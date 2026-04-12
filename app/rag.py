import os
import re
import time
import logging
import json
from collections import Counter
from io import StringIO

import pandas as pd
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingModel
from .db_client import ChromaClient

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL-SPECIFIC PROMPTS
#
# WHY SEPARATE PROMPTS?
# Gemini 2.5 Flash is a large frontier model — it handles nuanced, multi-rule
# instructions well and benefits from detailed guidance.
#
# Llama 3.2 1B and DeepSeek-R1 1.5B are tiny local models. Long, complex
# prompts confuse them and cause hallucination or instruction-following failure.
# They need a single, ultra-direct directive: "use only the context, answer
# the question, nothing else." This is not cheating the comparison — it's
# giving each model the prompt it can actually follow, which is the fair way
# to benchmark RAG faithfulness and accuracy.
#
# All three models receive IDENTICAL context chunks (same retrieval pipeline).
# The only difference is prompt style, not information.
# =============================================================================

# --- Gemini: detailed, nuanced, conversation-aware ---
GEMINI_PROMPT = """You are a precise AI assistant embedded in a knowledge retrieval system.

CONVERSATION HISTORY (most recent exchanges, for context continuity):
{history}

RETRIEVED DOCUMENT CONTEXT (from user's uploaded files):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Use the conversation history to understand follow-up questions and references like "it", "that", "the previous one".
- If the retrieved context is relevant, prefer it as your primary source of truth.
- If the context is empty or irrelevant, answer from conversation history or general knowledge and say so briefly.
- Do not explain your reasoning unless the user explicitly asks.
- Combine information across multiple document chunks and sources if needed.
- For numeric/dataset questions, return concrete values, not templates or SQL.
- Be direct and concise.
- Dont hide personal information."""

# --- Llama 3.2 1B: ultra-short, single directive ---
LLAMA_PROMPT = """Previous conversation:
{history}

Use ONLY the context below (and conversation above) to answer the question. If the context does not contain the answer, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

# --- DeepSeek-R1 1.5B ---
DEEPSEEK_PROMPT = """<history>
{history}
</history>

<context>
{context}
</context>

Answer the following question using the context and conversation history above.
If the answer is not in the context, say "Not found in documents."
Do not reason out loud. Give a direct answer only.
Use your Reasoning to generate answers.
Do not show thinking.
Dont hide personal information.


Question: {question}
Answer:"""

# --- Intent classification prompts (model-specific) ---
GEMINI_INTENT_PROMPT = """You are a query router. Classify the user query into exactly one category.

Available document sources the user has uploaded:
{sources}

Conversation history (last few turns):
{history}

User query: "{question}"

Rules:
- DATASET_QUERY: the query asks for numeric aggregation (max, min, average, sum, count, statistics) AND there is a relevant CSV/spreadsheet source that would contain that data.
- SMALLTALK: the query is a greeting, thanks, or casual chat with no information need.
- DOCUMENT_QUERY: everything else — questions about document content, explanations, summaries, descriptions, technical details.

Respond with exactly one word: DATASET_QUERY, SMALLTALK, or DOCUMENT_QUERY"""

LLAMA_INTENT_PROMPT = """Classify this query into one word: DATASET_QUERY, SMALLTALK, or DOCUMENT_QUERY.

Documents available: {sources}
Recent conversation: {history}
Query: "{question}"

DATASET_QUERY = asks for numbers/stats from a CSV file.
SMALLTALK = greeting or casual chat.
DOCUMENT_QUERY = anything else.

One word answer:"""

DEEPSEEK_INTENT_PROMPT = """<task>Classify the query. Reply with ONE word only.</task>

Documents: {sources}
History: {history}
Query: "{question}"

DATASET_QUERY = numeric stats from CSV
SMALLTALK = greeting/casual
DOCUMENT_QUERY = everything else

Answer:"""


def _format_history(history: list[dict]) -> str:
    """
    Convert a list of {"role": "user"|"assistant", "text": "..."} dicts
    into a compact human-readable string for prompt injection.
    Keeps only the last 6 turns (3 exchanges) to avoid context overflow.
    """
    if not history:
        return "(no previous conversation)"
    recent = history[-6:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['text'][:400]}")  # truncate very long turns
    return "\n".join(lines)


def _get_prompt(template: str, context: str, question: str, history: list[dict]) -> str:
    return template.format(
        context=context,
        question=question,
        history=_format_history(history),
    )


def _get_intent_prompt(model: str, sources: str, question: str, history: list[dict]) -> str:
    h = _format_history(history)
    if model == "gemini":
        return GEMINI_INTENT_PROMPT.format(sources=sources, question=question, history=h)
    elif model == "deepseek":
        return DEEPSEEK_INTENT_PROMPT.format(sources=sources, question=question, history=h)
    else:
        return LLAMA_INTENT_PROMPT.format(sources=sources, question=question, history=h)


INTENT_VALUES = {"DATASET_QUERY", "SMALLTALK", "DOCUMENT_QUERY"}


class RAGSystem:
    def __init__(self, embed_model: EmbeddingModel, db_client: ChromaClient, model="llama3.2:1b"):
        self.embed_model = embed_model
        self.db = db_client
        self.db.ensure_collection()
        self.local_llm = model

        # Per-user in-memory source cache (for retrieval rerouting within a session)
        self.recent_sources: dict = {}
        self.max_recent_sources = 15

        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # =========================================================================
    # CONVERSATION SOURCE CACHE
    # =========================================================================

    def get_recent_sources(self, google_id: str) -> list:
        return self.recent_sources.get(google_id, [])

    def update_recent_sources(self, google_id: str, source: str):
        sources = self.recent_sources.setdefault(google_id, [])
        if source not in sources:
            sources.append(source)
        self.recent_sources[google_id] = sources[-self.max_recent_sources:]

    # =========================================================================
    # DOCUMENT INDEXING
    # =========================================================================

    def index_documents(self, docs: list) -> dict:
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        embeddings = self.embed_model.embed_documents(texts)
        self.db.add_documents(ids, texts, metas, embeddings.tolist())
        print(f"Indexed {len(docs)} documents")
        return {"indexed": len(docs)}

    # =========================================================================
    # INTENT CLASSIFICATION  (LLM-based, model-aware, history-aware)
    # =========================================================================

    def classify_intent(self, question: str, google_id: str, model: str, history: list[dict]) -> str:
        """
        Ask the selected LLM to classify the query intent.
        Now receives conversation history so it can resolve references like
        "what about that CSV?" correctly.
        Returns one of: DATASET_QUERY | DOCUMENT_QUERY | SMALLTALK
        Falls back to DOCUMENT_QUERY on any failure.
        """
        try:
            collection = self.db.get_user_collection(google_id)
            res_all = collection.get()

            if res_all and "metadatas" in res_all:
                sources = list({
                    m.get("source", "")
                    for m in res_all["metadatas"]
                    if m.get("source")
                })
            else:
                sources = []

            source_summary = ", ".join(sources) if sources else "none"
            intent_prompt = _get_intent_prompt(model, source_summary, question, history)

            for attempt in range(2):
                raw = self._call_llm_raw(intent_prompt, model, max_tokens=10)
                if raw and raw.strip():
                    for intent in INTENT_VALUES:
                        if intent in raw.upper():
                            print(f"[Intent] '{intent}' classified by {model} | raw='{raw.strip()}'")
                            return intent
                    print(f"[Intent] Unrecognised response '{raw.strip()}' -> DOCUMENT_QUERY")
                    return "DOCUMENT_QUERY"
                if attempt == 0:
                    logger.warning("[Intent] Empty response on attempt 1, retrying...")
                    time.sleep(0.5)

            print("[Intent] No valid response after retries -> DOCUMENT_QUERY")
            return "DOCUMENT_QUERY"

        except Exception as e:
            logger.warning(f"Intent classification failed: {e} -> DOCUMENT_QUERY")
            return "DOCUMENT_QUERY"

    # =========================================================================
    # RAW LLM CALL  (intent classifier — no template wrapping)
    # =========================================================================

    def _call_llm_raw(self, prompt: str, model: str, max_tokens: int = 10) -> str:
        if model == "gemini":
            return self._gemini_raw(prompt, max_tokens)
        elif model == "deepseek":
            return self._ollama_raw(prompt, "deepseek-r1:1.5b", max_tokens)
        else:
            return self._ollama_raw(prompt, "llama3.2:1b-instruct-q4_K_M", max_tokens)

    def _gemini_raw(self, prompt: str, max_tokens: int = 10) -> str:
        if not self.api_key:
            return "DOCUMENT_QUERY"
        try:
            mdl = genai.GenerativeModel(self.model_name)
            resp = mdl.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
            )
            # Check if response was blocked due to safety
            if resp.candidates and resp.candidates[0].finish_reason == 2:  # SAFETY
                return "0.5"  # Default score for evaluation
            return resp.text.strip() if resp.text else "DOCUMENT_QUERY"
        except Exception as e:
            logger.warning(f"Gemini raw call failed: {e}")
            return "DOCUMENT_QUERY"

    def _ollama_raw(self, prompt: str, ollama_model: str, max_tokens: int = 10) -> str:
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.0}
                },
                timeout=120
            )
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content if content else "DOCUMENT_QUERY"
        except Exception as e:
            logger.warning(f"Ollama raw call failed: {e}")
            return "DOCUMENT_QUERY"

    # =========================================================================
    # JUDGE LLM CALL  (evaluation only — isolated from intent pipeline)
    #
    # WHY A SEPARATE METHOD?
    # _call_llm_raw is designed for intent classification: tiny token budget,
    # fallback of "DOCUMENT_QUERY", temperature 0. When used as a judge it
    # produces "DOCUMENT_QUERY" instead of a numeric score.
    #
    # This method is purpose-built for scoring: higher token budget so the
    # model can output a decimal, neutral fallback of "0.5", and a clean
    # single-turn stateless call with no caching or intent routing.
    # =========================================================================

    def _call_judge_llm(self, prompt: str, model: str) -> str:
        """
        Call an LLM as a judge to score answer quality.
        Returns a string containing a float between 0 and 1.
        Falls back to "0.5" (neutral) on any failure.

        Completely isolated from _call_llm_raw / intent pipeline.
        Uses a higher token budget (50) so models can output decimals reliably.
        """
        if model == "gemini":
            return self._gemini_judge(prompt)
        elif model == "deepseek":
            return self._ollama_judge(prompt, "deepseek-r1:1.5b")
        else:
            return self._ollama_judge(prompt, "llama3.2:1b-instruct-q4_K_M")

    def _gemini_judge(self, prompt: str) -> str:
        if not self.api_key:
            return "0.5"
        try:
            mdl = genai.GenerativeModel(self.model_name)
            resp = mdl.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=50)
            )
            if resp.candidates and resp.candidates[0].finish_reason == 2:  # SAFETY
                return "0.5"
            return resp.text.strip() if resp.text else "0.5"
        except Exception as e:
            logger.warning(f"Gemini judge call failed: {e}")
            return "0.5"

    def _ollama_judge(self, prompt: str, ollama_model: str) -> str:
        # DeepSeek-R1 is a thinking model: it emits <think>...</think> reasoning
        # tokens BEFORE the actual answer.  With a small num_predict budget the
        # model exhausts all tokens inside the think block and never reaches the
        # numeric score → every call returns "0.5" (the fallback).
        # Fix: use a generous budget (600) so the model can finish thinking AND
        # output the score, then strip the think block before returning.
        is_deepseek = "deepseek" in ollama_model.lower()
        num_predict = 600 if is_deepseek else 50
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "num_predict": num_predict,
                        "temperature": 0.0
                    }
                },
                timeout=180
            )
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()

            # Strip <think>...</think> blocks emitted by DeepSeek-R1 and similar
            # reasoning models before handing the text to the score parser.
            if is_deepseek and content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                # Also handle unclosed think blocks (model cut off mid-think):
                # if <think> appears but </think> never closed, the real answer
                # is likely after the last </think> — if there is none, the model
                # ran out of tokens inside the block; return raw so parser can try.
                if "<think>" in content and "</think>" not in content:
                    # Nothing after closing tag; return whatever is left so the
                    # regex parser in _parse_judge_score can still find a digit.
                    content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()

            return content if content else "0.5"
        except Exception as e:
            logger.warning(f"Ollama judge call failed: {e}")
            return "0.5"

    # =========================================================================
    # RETRIEVAL — multi-source aware with history-aware follow-up handling
    #
    # Key change from previous version:
    # Instead of picking ONE best source and only returning its chunks, we now:
    #   1. Run a global vector search across ALL user chunks.
    #   2. Identify ALL sources whose best chunk clears the similarity threshold.
    #   3. For each qualifying source, fetch its top-k chunks.
    #   4. Merge and return — the LLM receives context from multiple documents.
    #
    # NEW: If this is a follow-up query (detected via history), prioritize
    # retrieval from recent sources to maintain conversation context.
    #
    # This means a question like "compare the Q1 report and the sales CSV" will
    # pull relevant chunks from both files rather than arbitrarily choosing one.
    # And "ok but just give me topic names" will retrieve from the same document
    # as the previous query if it's contextually relevant.
    # =========================================================================

    def retrieve(self, query: str, google_id: str, k: int = 6, history: list[dict] = None) -> list:
        if history is None:
            history = []
        
        q_emb = self.embed_model.embed_query(query)
        collection = self.db.get_user_collection(google_id)

        # --- 1. Follow-up query detection + recent-source priority cache ---
        # If this is a follow-up query (probabilistically detected) and we have
        # recent sources in cache, try to retrieve from those first to maintain context.
        is_followup = self._is_followup_query(query, history=history)
        recent_sources = self.get_recent_sources(google_id)
        
        if is_followup and recent_sources:
            print(f"[Retrieval] Follow-up detected. Prioritizing recent sources: {recent_sources}")
            try:
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(k * 2, 20),
                    where={"source": {"$in": recent_sources}}
                )
                docs, best_sim = self._parse_query_results(res, threshold=0.25)
                if docs and best_sim >= 0.35:
                    print(f"[Retrieval] Follow-up context hit — retrieved {len(docs)} chunks from recent sources")
                    return docs
                else:
                    print(f"[Retrieval] Follow-up query but low similarity (best_sim={best_sim:.3f}) — falling back to global search")
            except Exception as e:
                logger.warning(f"Follow-up cache retrieval failed: {e}")
        elif recent_sources and not is_followup:
            # Non-follow-up query with recent sources: try cache but be lenient
            try:
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(k * 2, 20),
                    where={"source": {"$in": recent_sources}}
                )
                docs, best_sim = self._parse_query_results(res, threshold=0.25)
                if docs and best_sim >= 0.35:
                    print(f"[Retrieval] Cache hit — sources: {list({d['metadata']['source'] for d in docs})}")
                    return docs
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        # --- 2. Filename token match ---
        res_all = collection.get()
        if not res_all or "metadatas" not in res_all:
            return []

        all_sources = list({
            m.get("source") for m in res_all["metadatas"] if m.get("source")
        })

        query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        filename_matched_docs = []
        for source in all_sources:
            name_words = set(re.sub(r"[._\-]", " ", source.lower()).split())
            overlap = {w for w in (query_words & name_words) if len(w) > 2}
            if overlap:
                print(f"[Retrieval] Filename match: '{source}' overlap={overlap}")
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": source}
                )
                docs, _ = self._parse_query_results(res, threshold=0.10)
                filename_matched_docs.extend(docs)

        if filename_matched_docs:
            return filename_matched_docs

        # --- 3. Global vector search → identify ALL qualifying sources ---
        # Fetch a generous pool of results so we can score every source fairly.
        pool_size = min(len(res_all.get("ids", [])), 200)
        if pool_size == 0:
            return []

        res = collection.query(query_embeddings=[q_emb], n_results=pool_size)
        if not res or "documents" not in res:
            return []

        all_hits, _ = self._parse_query_results(res, threshold=0.15)
        if not all_hits:
            return []

        # Best similarity score per source
        source_best: dict[str, float] = {}
        for hit in all_hits:
            src = hit["metadata"].get("source", "")
            source_best[src] = max(source_best.get(src, 0.0), hit["score"])

        RELEVANCE_THRESHOLD = 0.30
        # Qualify every source that clears the threshold (not just the single best)
        qualifying_sources = [
            src for src, score in source_best.items()
            if score >= RELEVANCE_THRESHOLD
        ]

        if not qualifying_sources:
            # Nothing clears the threshold — return top-k global hits anyway
            print("[Retrieval] No source cleared threshold, returning top-k global hits")
            return sorted(all_hits, key=lambda x: x["score"], reverse=True)[:k]

        print(f"[Retrieval] Qualifying sources: {qualifying_sources}")

        # --- 4. For each qualifying source, fetch its top-k chunks ---
        # Distribute k evenly across sources (at least 2 chunks per source).
        chunks_per_source = max(2, k // len(qualifying_sources))
        merged_docs = []

        for src in qualifying_sources:
            self.update_recent_sources(google_id, src)
            try:
                res_src = collection.query(
                    query_embeddings=[q_emb],
                    n_results=chunks_per_source,
                    where={"source": src}
                )
                src_docs, _ = self._parse_query_results(res_src, threshold=0.15)
                # Sort by chunk_index for coherent reading order
                src_docs.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
                merged_docs.extend(src_docs)
            except Exception as e:
                logger.warning(f"Per-source retrieval failed for '{src}': {e}")

        return merged_docs

    # =========================================================================
    # FOLLOW-UP QUERY DETECTION — Probabilistic Semantic Similarity
    # =========================================================================

    def _is_followup_query(self, question: str, history: list[dict] = None) -> bool:
        """
        Probabilistic follow-up detection using semantic similarity and query characteristics.
        
        Instead of hardcoded regex patterns, this uses:
        1. **Embedding Similarity**: How similar is this query to the previous query?
        2. **Query Length**: Shorter queries are more likely follow-ups (refinement/filtering)
        3. **History Presence**: With history + short query = likely follow-up
        4. **Context Reuse Score**: Does the query reuse words from history?
        
        Returns a boolean: True if query is detected as a follow-up with confidence > threshold.
        
        Heuristics:
        - High similarity to previous query: Strong indicator (refinement/elaboration)
        - Short query + history: Likely follow-up (user narrowing down)
        - Moderate similarity + low information content: Follow-up pattern
        """
        if history is None or len(history) < 2:
            return False
        
        # Extract previous user query from history
        prev_queries = [msg.get("text", "") for msg in history if msg.get("role") == "user"]
        if not prev_queries:
            return False
        
        prev_query = prev_queries[-1]
        
        # ==== Heuristic 1: Embedding Similarity ====
        # High semantic similarity to previous query suggests follow-up
        try:
            current_emb = self.embed_model.embed_query(question)
            prev_emb = self.embed_model.embed_query(prev_query)
            sim_score = cosine_similarity([current_emb], [prev_emb])[0][0]
        except Exception as e:
            logger.warning(f"Embedding similarity computation failed: {e}")
            sim_score = 0.0
        
        # If very similar to previous query, it's likely a follow-up
        if sim_score >= 0.45:
            print(f"[FollowUp] High semantic similarity ({sim_score:.3f}) → Follow-up detected")
            return True
        
        # ==== Heuristic 2: Query Length ====
        # Follow-ups tend to be shorter (user refining/narrowing)
        current_tokens = len(question.split())
        prev_tokens = len(prev_query.split())
        
        # Short query (≤ 7 tokens) + history = likely refinement follow-up
        is_short = current_tokens <= 7
        is_significantly_shorter = current_tokens <= (prev_tokens * 0.6 + 2)  # 60% of previous length
        
        # ==== Heuristic 3: Context Reuse (Word Overlap) ====
        # Does current query reuse words from recent history?
        current_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        prev_words = set(re.sub(r"[^\w\s]", "", prev_query.lower()).split())
        
        # Remove common stop-words for cleaner overlap detection
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "be", "have", "do",
            "what", "how", "when", "where", "why", "who", "can", "could", "would",
            "should", "will", "might", "must", "may", "you", "i", "we", "me"
        }
        
        meaningful_current = {w for w in current_words if len(w) > 2 and w not in stop_words}
        meaningful_prev = {w for w in prev_words if len(w) > 2 and w not in stop_words}
        
        word_overlap = meaningful_current & meaningful_prev
        overlap_ratio = len(word_overlap) / max(len(meaningful_current), 1)  # % of current words in previous
        
        # ==== Decision Logic (Probabilistic) ====
        
        # Case 1: Moderate-to-high similarity + short query = Follow-up
        if sim_score >= 0.35 and is_short:
            print(f"[FollowUp] Moderate similarity ({sim_score:.3f}) + short query ({current_tokens} tokens) → Follow-up detected")
            return True
        
        # Case 2: Short query + high word overlap + history = Follow-up
        if is_short and overlap_ratio >= 0.4:
            print(f"[FollowUp] Short query + high context reuse ({overlap_ratio:.2%}) → Follow-up detected")
            return True
        
        # Case 3: Significantly shorter query + moderate overlap = Follow-up
        if is_significantly_shorter and overlap_ratio >= 0.25 and sim_score >= 0.25:
            print(f"[FollowUp] Query refined (len factor: {(current_tokens/prev_tokens):.1%}) + context reuse → Follow-up detected")
            return True
        
        # Case 4: Very short follow-up refinement patterns like "just names", "in csv", "top 5"
        if current_tokens <= 5:
            # Very short phrase with some context overlap is likely a follow-up refinement
            if overlap_ratio >= 0.15 or word_overlap:
                print(f"[FollowUp] Minimal query ({current_tokens} tokens) + any context reuse → Follow-up detected")
                return True
        
        return False


    def _parse_query_results(self, res: dict, threshold: float) -> tuple:
        docs = []
        best_sim = 0.0
        if not res or "documents" not in res:
            return docs, best_sim
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            sim = 1 - (dist / 2)
            best_sim = max(best_sim, sim)
            if sim >= threshold:
                docs.append({"text": doc, "metadata": meta, "score": sim})
        return docs, best_sim

    # =========================================================================
    # LLM CALLS  (all now accept history)
    # =========================================================================

    def call_gemini(self, question: str, context: str, history: list[dict], max_tokens: int = 1000) -> str:
        if not self.api_key:
            return "Gemini API key missing."
        prompt = _get_prompt(GEMINI_PROMPT, context, question, history)
        mdl = genai.GenerativeModel(self.model_name)
        for attempt in range(3):
            try:
                response = mdl.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
                )
                return response.text
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Gemini rate limit — waiting 10s (attempt {attempt+1})")
                    time.sleep(10)
                    continue
                logger.error(f"Gemini API error: {e}")
                return f"Error calling Gemini API: {e}"
        return "Gemini rate limit exceeded. Please try again later."

    def call_ollama(self, question: str, context: str, history: list[dict], model: str = "ollama") -> str:
        if model == "deepseek":
            ollama_model = "deepseek-r1:1.5b"
            prompt = _get_prompt(DEEPSEEK_PROMPT, context, question, history)
        else:
            ollama_model = "llama3.2:1b-instruct-q4_K_M"
            prompt = _get_prompt(LLAMA_PROMPT, context, question, history)

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "num_ctx": 2000,
                        "num_predict": 1000,
                        "temperature": 0.1
                    }
                },
                timeout=600
            )
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "").strip()
            thinking = message.get("thinking", "").strip()
            return content if content else thinking
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Ollama error: {e}"

    def generate(self, question: str, context: str, history: list[dict], model: str = "gemini") -> str:
        """Route to the right LLM with its model-specific prompt."""
        if model == "gemini":
            return self.call_gemini(question, context, history)
        elif model == "deepseek":
            return self.call_ollama(question, context, history, model="deepseek")
        elif model.startswith("llama") or model in ("ollama", "llama"):
            return self.call_ollama(question, context, history, model="ollama")
        return "Unknown model selected."

    # =========================================================================
    # CSV / DATASET HANDLING
    # =========================================================================

    def find_and_load_best_csv(self, question: str, google_id: str) -> tuple:
        """
        Find the most semantically relevant CSV for the query.
        Returns (DataFrame, source_name) or (None, None).
        """
        collection = self.db.get_user_collection(google_id)
        res_all = collection.get()

        if not res_all or "metadatas" not in res_all:
            return None, None

        csv_sources = list({
            m["source"]
            for m in res_all["metadatas"]
            if m.get("source", "").lower().endswith(".csv")
        })

        if not csv_sources:
            return None, None

        q_emb = self.embed_model.embed_query(question)
        source_scores: dict = {}

        for src in csv_sources:
            try:
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=5,
                    where={"source": src}
                )
                if res and "distances" in res and res["distances"][0]:
                    best_sim = max(1 - (d / 2) for d in res["distances"][0])
                    source_scores[src] = best_sim
                    print(f"[CSV] {src} -> score={best_sim:.3f}")
            except Exception as e:
                logger.warning(f"CSV scoring failed for {src}: {e}")

        if not source_scores:
            return None, None

        best_source = max(source_scores, key=source_scores.get)
        best_score = source_scores[best_source]

        if best_score < 0.25:
            print(f"[CSV] Best CSV '{best_source}' score={best_score:.3f} below threshold — skipping")
            return None, None

        print(f"[CSV] Selected: '{best_source}' (score={best_score:.3f})")

        try:
            source_data = collection.get(where={"source": best_source})
            if not source_data or "documents" not in source_data:
                return None, best_source

            raw_text = "\n".join(source_data["documents"])

            try:
                df = pd.read_csv(StringIO(raw_text), low_memory=False)
                if len(df.columns) == 1:
                    raise ValueError("Single-column fallback")
            except Exception:
                lines = [
                    re.sub(r"\s{2,}", ",", l.strip())
                    for l in raw_text.split("\n") if l.strip()
                ]
                df = pd.read_csv(StringIO("\n".join(lines)))

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")

            return df, best_source

        except Exception as e:
            logger.warning(f"CSV load failed for {best_source}: {e}")
            return None, best_source

    def build_dataset_context(self, df, source: str) -> str:
        """Build a compact, LLM-readable statistical summary."""
        import numpy as np

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        rows, cols = df.shape
        parts = [f"DATASET: {source}", f"Shape: {rows} rows x {cols} columns", ""]

        for col in df.columns:
            if col in numeric_cols:
                s = df[col].describe()
                parts.append(
                    f"Column '{col}' (numeric) | count={int(s['count'])} "
                    f"min={s['min']:.2f} max={s['max']:.2f} "
                    f"mean={s['mean']:.2f} median={s['50%']:.2f} std={s['std']:.2f}"
                )
            else:
                parts.append(
                    f"Column '{col}' (text) | unique={df[col].nunique()} "
                    f"sample={df[col].dropna().head(3).tolist()}"
                )

        parts.append("\nSAMPLE ROWS (first 5):")
        parts.append(df.head(5).to_csv(index=False))

        if len(numeric_cols) > 1:
            parts.append("CORRELATIONS:")
            parts.append(df[numeric_cols].corr().round(3).to_string())

        return "\n".join(parts)

    # =========================================================================
    # MAIN RAG PIPELINE  — now history-aware + multi-source
    # =========================================================================

    def answer(self, question: str, google_id: str, top_k: int, model: str = "gemini",
               history: list[dict] = None) -> dict:
        """
        history: list of {"role": "user"|"assistant", "text": "..."} dicts,
                 ordered oldest→newest. Passed in from MongoDB by the /query endpoint.
        """
        if history is None:
            history = []

        if not question or not question.strip():
            return {"answer": "Question cannot be empty.", "sources": [], "retrieved_count": 0, "eval": {}}

        pipeline_start = time.time()

        # ------------------------------------------------------------------
        # STEP 1: LLM-based intent classification (now history-aware)
        # ------------------------------------------------------------------
        intent_start = time.time()
        intent = self.classify_intent(question, google_id, model, history)
        intent_latency = time.time() - intent_start
        print(f"[Pipeline] Intent={intent} ({intent_latency:.2f}s via {model})")

        # ------------------------------------------------------------------
        # STEP 2: Route by intent
        # ------------------------------------------------------------------
        if intent == "SMALLTALK":
            gen_start = time.time()
            answer_text = self.generate(question, "", history, model)
            gen_latency = time.time() - gen_start
            return self._build_response(
                answer=answer_text, sources=[], retrieved_count=0,
                context_chars=0, intent=intent, model=model,
                intent_latency=intent_latency, gen_latency=gen_latency,
                total_latency=time.time() - pipeline_start, context=""
            )

        if intent == "DATASET_QUERY":
            df, csv_source = self.find_and_load_best_csv(question, google_id)
            if df is not None and not df.empty:
                context = self.build_dataset_context(df, csv_source)
                gen_start = time.time()
                answer_text = self.generate(question, context, history, model)
                gen_latency = time.time() - gen_start
                return self._build_response(
                    answer=answer_text,
                    sources=[{"filename": csv_source, "snippet": ""}],
                    retrieved_count=0, context_chars=len(context),
                    intent=intent, model=model,
                    intent_latency=intent_latency, gen_latency=gen_latency,
                    total_latency=time.time() - pipeline_start, context=context
                )
            print("[Pipeline] DATASET_QUERY but no relevant CSV — falling back to document retrieval")
            intent = "DOCUMENT_QUERY"

        # ------------------------------------------------------------------
        # STEP 3: Multi-source document retrieval (now history-aware)
        # ------------------------------------------------------------------
        ret_start = time.time()
        retrieved_docs = self.retrieve(question, google_id, k=top_k, history=history)
        ret_latency = time.time() - ret_start
        print(f"[Pipeline] Retrieved {len(retrieved_docs)} chunks ({ret_latency:.2f}s)")

        if not retrieved_docs:
            gen_start = time.time()
            answer_text = self.generate(question, "", history, model)
            gen_latency = time.time() - gen_start
            return self._build_response(
                answer=answer_text, sources=[], retrieved_count=0,
                context_chars=0, intent=intent, model=model,
                intent_latency=intent_latency, gen_latency=gen_latency,
                total_latency=time.time() - pipeline_start,
                retrieval_latency=ret_latency, context=""
            )

        # ------------------------------------------------------------------
        # STEP 4: Build multi-source context
        #
        # Group chunks by source, sort each group by chunk_index for
        # coherent reading order, then concatenate with clear source headers.
        # This way the LLM knows which text came from which file.
        # ------------------------------------------------------------------
        from collections import defaultdict
        source_chunks: dict = defaultdict(list)
        for d in retrieved_docs:
            src = d["metadata"].get("source", "unknown")
            source_chunks[src].append(d)

        # Sort each source's chunks by chunk_index
        for src in source_chunks:
            source_chunks[src].sort(key=lambda x: x["metadata"].get("chunk_index", 0))

        # Build context block with clear per-source headers
        context_parts = []
        sources_out = []
        for src, chunks in source_chunks.items():
            context_parts.append(f"=== Source: {src} ===")
            context_parts.append("\n".join(c["text"] for c in chunks))
            sources_out.append({
                "filename": src,
                "snippet": chunks[0]["text"][:200] if chunks else ""
            })

        context = "\n\n".join(context_parts)
        print(f"[Pipeline] Context: {len(retrieved_docs)} chunks from {len(source_chunks)} source(s): "
              f"{list(source_chunks.keys())}")

        # ------------------------------------------------------------------
        # STEP 5: Generate with history + multi-source context
        # ------------------------------------------------------------------
        gen_start = time.time()
        answer_text = self.generate(question, context, history, model)
        gen_latency = time.time() - gen_start

        return self._build_response(
            answer=answer_text, sources=sources_out,
            retrieved_count=len(retrieved_docs), context_chars=len(context),
            intent=intent, model=model,
            intent_latency=intent_latency, gen_latency=gen_latency,
            total_latency=time.time() - pipeline_start,
            retrieval_latency=ret_latency, context=context
        )

    # =========================================================================
    # RESPONSE BUILDER
    # =========================================================================

    def _build_response(
        self,
        answer: str,
        sources: list,
        retrieved_count: int,
        context_chars: int,
        intent: str,
        model: str,
        intent_latency: float,
        gen_latency: float,
        total_latency: float,
        retrieval_latency: float = 0.0,
        context: str = "",
    ) -> dict:
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": retrieved_count,
            "context": context,  # Include context for evaluation
            "eval": {
                "model": model,
                "intent": intent,
                "context_chars": context_chars,
                "retrieved_count": retrieved_count,
                "latency": {
                    "intent_classification_s": round(intent_latency, 3),
                    "retrieval_s": round(retrieval_latency, 3),
                    "generation_s": round(gen_latency, 3),
                    "total_s": round(total_latency, 3),
                },
            }
        }
