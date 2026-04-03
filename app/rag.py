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

# --- Gemini: detailed, nuanced ---
GEMINI_PROMPT = """You are a precise AI assistant embedded in a knowledge retrieval system.

CONTEXT (retrieved from user's documents):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- If the context is relevant, answer ONLY using information from it.
- If the context is empty or irrelevant, answer from general knowledge and say so briefly.
- Do not explain your reasoning unless the user explicitly asks.
- Combine information across chunks if needed for a complete answer.
- For numeric/dataset questions, return concrete values, not templates or SQL.
- Be direct and concise."""

# --- Llama 3.2 1B: ultra-short, single directive ---
# Small models hallucinate when given long instruction lists.
# One clear rule outperforms five nuanced ones at 1B scale.
LLAMA_PROMPT = """Use ONLY the context below to answer the question. Do not add anything not in the context. If the context does not contain the answer, say "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

# --- DeepSeek-R1 1.5B: same philosophy as llama, slight structural difference
# to match DeepSeek's instruction-following training format ---
DEEPSEEK_PROMPT = """<context>
{context}
</context>

Answer the following question using ONLY the information in the context above.
If the answer is not in the context, say "Not found in documents."
Do not reason out loud. Give a direct answer only.

Question: {question}
Answer:"""

# --- Intent classification prompts (model-specific) ---
# The classifier must return a single word. Small models fail at JSON or
# multi-line outputs, so we keep the format as simple as possible.

GEMINI_INTENT_PROMPT = """You are a query router. Classify the user query into exactly one category.

Available document sources the user has uploaded:
{sources}

User query: "{question}"

Rules:
- DATASET_QUERY: the query asks for numeric aggregation (max, min, average, sum, count, statistics) AND there is a relevant CSV/spreadsheet source that would contain that data.
- SMALLTALK: the query is a greeting, thanks, or casual chat with no information need.
- DOCUMENT_QUERY: everything else — questions about document content, explanations, summaries, descriptions, technical details.

Respond with exactly one word: DATASET_QUERY, SMALLTALK, or DOCUMENT_QUERY"""

LLAMA_INTENT_PROMPT = """Classify this query into one word: DATASET_QUERY, SMALLTALK, or DOCUMENT_QUERY.

Documents available: {sources}
Query: "{question}"

DATASET_QUERY = asks for numbers/stats from a CSV file.
SMALLTALK = greeting or casual chat.
DOCUMENT_QUERY = anything else.

One word answer:"""

DEEPSEEK_INTENT_PROMPT = """<task>Classify the query. Reply with ONE word only.</task>

Documents: {sources}
Query: "{question}"

DATASET_QUERY = numeric stats from CSV
SMALLTALK = greeting/casual
DOCUMENT_QUERY = everything else

Answer:"""


def _get_prompt(template: str, context: str, question: str) -> str:
    return template.format(context=context, question=question)


def _get_intent_prompt(model: str, sources: str, question: str) -> str:
    if model == "gemini":
        return GEMINI_INTENT_PROMPT.format(sources=sources, question=question)
    elif model == "deepseek":
        return DEEPSEEK_INTENT_PROMPT.format(sources=sources, question=question)
    else:
        return LLAMA_INTENT_PROMPT.format(sources=sources, question=question)


INTENT_VALUES = {"DATASET_QUERY", "SMALLTALK", "DOCUMENT_QUERY"}


class RAGSystem:
    def __init__(self, embed_model: EmbeddingModel, db_client: ChromaClient, model="llama3.2:1b"):
        self.embed_model = embed_model
        self.db = db_client
        self.db.ensure_collection()
        self.local_llm = model

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
    # INTENT CLASSIFICATION  (LLM-based, model-aware)
    # =========================================================================

    def classify_intent(self, question: str, google_id: str, model: str) -> str:
        """
        Ask the selected LLM to classify the query intent.
        Returns one of: DATASET_QUERY | DOCUMENT_QUERY | SMALLTALK

        The model sees what document sources exist so it can make an informed
        decision — e.g. it won't classify as DATASET_QUERY if only PDFs exist.

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
            intent_prompt = _get_intent_prompt(model, source_summary, question)

            # Retry once on empty response
            for attempt in range(2):
                raw = self._call_llm_raw(intent_prompt, model, max_tokens=10)
                
                if raw and raw.strip():  # Non-empty response
                    for intent in INTENT_VALUES:
                        if intent in raw.upper():
                            print(f"[Intent] '{intent}' classified by {model} | raw='{raw.strip()}'")
                            return intent
                    print(f"[Intent] Unrecognised response '{raw.strip()}' -> DOCUMENT_QUERY")
                    return "DOCUMENT_QUERY"
                
                # Empty response, retry once
                if attempt == 0:
                    logger.warning(f"[Intent] Empty response on attempt 1, retrying...")
                    time.sleep(0.5)
                    continue

            # Both attempts failed or returned empty
            print(f"[Intent] No valid response after retries -> DOCUMENT_QUERY")
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
            return self._ollama_raw(prompt, "llama3.2:1b", max_tokens)

    def _gemini_raw(self, prompt: str, max_tokens: int = 10) -> str:
        if not self.api_key:
            return "DOCUMENT_QUERY"
        try:
            mdl = genai.GenerativeModel(self.model_name)
            resp = mdl.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
            )
            # Handle empty response (finish_reason 2 = STOP with no content)
            if resp.text:
                return resp.text.strip()
            return "DOCUMENT_QUERY"
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
    # RETRIEVAL
    # =========================================================================

    def retrieve(self, query: str, google_id: str, k: int = 6) -> list:
        q_emb = self.embed_model.embed_query(query)
        collection = self.db.get_user_collection(google_id)

        # --- Recent source cache ---
        recent_sources = self.get_recent_sources(google_id)
        print(f"\n[Retrieval] Recent sources in cache: {recent_sources}")

        if recent_sources:
            try:
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": {"$in": recent_sources}}
                )
                docs, best_sim = self._parse_query_results(res, threshold=0.25)
                if docs and best_sim >= 0.30:
                    print("[Retrieval] Cache hit — using recent source")
                    return docs
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        # --- All sources ---
        res_all = collection.get()
        if not res_all or "metadatas" not in res_all:
            return []

        sources = list({
            m.get("source")
            for m in res_all["metadatas"]
            if m.get("source")
        })

        # --- Filename token match (meaningful words only, length > 2) ---
        query_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        for source in sources:
            name_words = set(re.sub(r"[._\-]", " ", source.lower()).split())
            meaningful_overlap = {w for w in (query_words & name_words) if len(w) > 2}
            if meaningful_overlap:
                print(f"[Retrieval] Filename match: '{source}' overlap={meaningful_overlap}")
                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=k,
                    where={"source": source}
                )
                docs, _ = self._parse_query_results(res, threshold=0.10)
                if docs:
                    return docs

        # --- Global vector search -> best source reroute ---
        res = collection.query(query_embeddings=[q_emb], n_results=100)
        if not res or "documents" not in res:
            return []

        all_hits, _ = self._parse_query_results(res, threshold=0.15)
        if not all_hits:
            return []

        source_best: dict = {}
        for hit in all_hits:
            src = hit["metadata"].get("source", "")
            source_best[src] = max(source_best.get(src, 0), hit["score"])

        best_source = max(source_best, key=source_best.get)
        best_score = source_best[best_source]
        print(f"[Retrieval] Global best: '{best_source}' score={best_score:.3f}")

        if best_score < 0.30:
            return sorted(all_hits, key=lambda x: x["score"], reverse=True)[:k]

        res_source = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where={"source": best_source}
        )
        docs, _ = self._parse_query_results(res_source, threshold=0.15)
        return docs

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
    # LLM CALLS
    # =========================================================================

    def call_gemini(self, question: str, context: str, max_tokens: int = 1000) -> str:
        if not self.api_key:
            return "Gemini API key missing."
        prompt = _get_prompt(GEMINI_PROMPT, context, question)
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

    def call_ollama(self, question: str, context: str, model: str = "ollama") -> str:
        if model == "deepseek":
            ollama_model = "deepseek-r1:1.5b"
            prompt = _get_prompt(DEEPSEEK_PROMPT, context, question)
        else:
            ollama_model = "llama3.2:1b-instruct-q4_K_M"
            prompt = _get_prompt(LLAMA_PROMPT, context, question)

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

    def generate(self, question: str, context: str, model: str = "gemini") -> str:
        """Route to the right LLM with its model-specific prompt."""
        if model == "gemini":
            return self.call_gemini(question, context)
        elif model == "deepseek":
            return self.call_ollama(question, context, model="deepseek")
        elif model in ("ollama", "llama"):
            return self.call_ollama(question, context, model="ollama")
        return "Unknown model selected."

    # =========================================================================
    # CSV / DATASET HANDLING
    # =========================================================================

    def find_and_load_best_csv(self, question: str, google_id: str) -> tuple:
        """
        Find the most semantically relevant CSV for the query, not just any CSV.
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

    def build_dataset_context(self, df: pd.DataFrame, source: str) -> str:
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
    # MAIN RAG PIPELINE
    # =========================================================================

    def answer(self, question: str, google_id: str, top_k: int, model: str = "gemini") -> dict:
        if not question or not question.strip():
            return {
                "answer": "Question cannot be empty.",
                "sources": [],
                "retrieved_count": 0,
                "eval": {}
            }

        pipeline_start = time.time()

        # ------------------------------------------------------------------
        # STEP 1: LLM-based intent classification
        # ------------------------------------------------------------------
        intent_start = time.time()
        intent = self.classify_intent(question, google_id, model)
        intent_latency = time.time() - intent_start
        print(f"[Pipeline] Intent={intent} ({intent_latency:.2f}s via {model})")

        # ------------------------------------------------------------------
        # STEP 2: Route by intent
        # ------------------------------------------------------------------

        if intent == "SMALLTALK":
            gen_start = time.time()
            answer_text = self.generate(question, "", model)
            gen_latency = time.time() - gen_start
            return self._build_response(
                answer=answer_text, sources=[], retrieved_count=0,
                context_chars=0, intent=intent, model=model,
                intent_latency=intent_latency, gen_latency=gen_latency,
                total_latency=time.time() - pipeline_start
            )

        if intent == "DATASET_QUERY":
            df, csv_source = self.find_and_load_best_csv(question, google_id)
            if df is not None and not df.empty:
                context = self.build_dataset_context(df, csv_source)
                gen_start = time.time()
                answer_text = self.generate(question, context, model)
                gen_latency = time.time() - gen_start
                return self._build_response(
                    answer=answer_text,
                    sources=[{"filename": csv_source, "snippet": ""}],
                    retrieved_count=0, context_chars=len(context),
                    intent=intent, model=model,
                    intent_latency=intent_latency, gen_latency=gen_latency,
                    total_latency=time.time() - pipeline_start
                )
            # No relevant CSV — fall through to document retrieval
            print("[Pipeline] DATASET_QUERY but no relevant CSV — falling back to document retrieval")
            intent = "DOCUMENT_QUERY"

        # ------------------------------------------------------------------
        # STEP 3: Document retrieval
        # ------------------------------------------------------------------
        ret_start = time.time()
        retrieved_docs = self.retrieve(question, google_id, k=top_k)
        ret_latency = time.time() - ret_start
        print(f"[Pipeline] Retrieved {len(retrieved_docs)} chunks ({ret_latency:.2f}s)")

        if not retrieved_docs:
            gen_start = time.time()
            answer_text = self.generate(question, "", model)
            gen_latency = time.time() - gen_start
            return self._build_response(
                answer=answer_text, sources=[], retrieved_count=0,
                context_chars=0, intent=intent, model=model,
                intent_latency=intent_latency, gen_latency=gen_latency,
                total_latency=time.time() - pipeline_start,
                retrieval_latency=ret_latency
            )

        # ------------------------------------------------------------------
        # STEP 4: Select best source, build shared context
        # ------------------------------------------------------------------
        source_counts = Counter(
            d["metadata"].get("source", "unknown") for d in retrieved_docs
        )
        best_source = source_counts.most_common(1)[0][0]
        self.update_recent_sources(google_id, best_source)

        docs = sorted(
            [d for d in retrieved_docs if d["metadata"].get("source") == best_source],
            key=lambda x: x["metadata"].get("chunk_index", 0)
        )

        # All models get the SAME context — fair comparison
        context = f"Source document: {best_source}\n\n" + "\n\n".join(d["text"] for d in docs)
        print(f"[Pipeline] Context: {len(docs)} chunks, {len(context)} chars from '{best_source}'")

        # ------------------------------------------------------------------
        # STEP 5: Generate
        # ------------------------------------------------------------------
        gen_start = time.time()
        answer_text = self.generate(question, context, model)
        gen_latency = time.time() - gen_start

        sources = [{
            "filename": best_source,
            "snippet": docs[0]["text"][:200] if docs else ""
        }]

        return self._build_response(
            answer=answer_text, sources=sources,
            retrieved_count=len(docs), context_chars=len(context),
            intent=intent, model=model,
            intent_latency=intent_latency, gen_latency=gen_latency,
            total_latency=time.time() - pipeline_start,
            retrieval_latency=ret_latency
        )

    # =========================================================================
    # RESPONSE BUILDER — eval metadata attached to every response
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
    ) -> dict:
        """
        Attaches an 'eval' block to every response so your frontend/logger
        can record and compare models across latency, context size, and intent.
        """
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": retrieved_count,
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