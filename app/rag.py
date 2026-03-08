import os
import logging
from collections import defaultdict, Counter
from dotenv import load_dotenv
import time
import requests

import google.generativeai as genai

from .embeddings import EmbeddingModel
from .db_client import ChromaClient

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant answering questions using a document knowledge base.

Rules:
- Use ONLY the information from the provided context.
- Do NOT invent information.
- Extract the actual explanations from the text.
- Do NOT describe document structure (page numbers, section titles, etc).
- Focus on explaining the concept asked by the user.
- Combine relevant information across chunks to produce a full explanation.

If the context contains multiple parts of an explanation,
merge them together.

If the answer is not present in the context say:
"The documents do not contain this information."

Response format:

## Answer
Clear explanation.

## Key Points
- bullet
- bullet
- bullet

## Sources
Document names used.

CONTEXT:
{context}

QUESTION:
{question}
"""


class RAGSystem:
    def __init__(self, embed_model: EmbeddingModel, db_client: ChromaClient):
        self.embed_model = embed_model
        self.db = db_client
        self.db.ensure_collection()

        self.api_key = os.getenv("GEMINI_API_KEY")

        if self.api_key:
            genai.configure(api_key=self.api_key)

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # -------------------------
    # DOCUMENT INDEXING
    # -------------------------

    def index_documents(self, docs):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]

        embeddings = self.embed_model.embed_documents(texts)

        self.db.add_documents(ids, texts, metas, embeddings.tolist())

        logger.info(f"Indexed {len(docs)} documents")

        return {"indexed": len(docs)}

    # -------------------------
    # RETRIEVAL
    # -------------------------

    def retrieve(self, query, google_id, k=6):

        q_emb = self.embed_model.embed_query(query)

        collection = self.db.get_user_collection(google_id)

        # Get all document sources
        res_all = collection.get()

        if not res_all or "metadatas" not in res_all:
            return []

        sources = list({
            m.get("source")
            for m in res_all["metadatas"]
            if m.get("source")
        })

        if not sources:
            return []

        # -------------------------
        # DOCUMENT ROUTING
        # -------------------------

        best_source = None
        best_score = -1
        best_docs = []

        for source in sources:

            try:

                res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=1,
                    where={"source": source}
                )

                if res and res.get("distances"):

                    score = 1 - res["distances"][0][0]  # similarity

                    if score > best_score:
                        best_score = score
                        best_source = source
                        best_docs = res

            except Exception as e:
                logger.warning(f"Routing check failed for {source}: {e}")

        if not best_source:
            return []

        logger.info(f"Selected document: {best_source}")

        # -------------------------
        # FINAL RETRIEVAL
        # -------------------------

        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            where={"source": best_source}
        )

        docs = []

        if res and "documents" in res:

            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                docs.append({
                    "text": doc,
                    "metadata": meta
                })

        logger.info(f"Retrieved {len(docs)} chunks from {best_source}")

        return docs
    # -------------------------
    # CONTEXT EXPANSION
    # -------------------------

    def expand_context(self, docs, google_id, max_chunks_per_doc=3):

        collection = self.db.get_user_collection(google_id)

        sources = set(
            d["metadata"].get("source")
            for d in docs
        )

        expanded_docs = []

        for source in sources:

            try:

                res = collection.get(where={"source": source})

                if not res or "documents" not in res:
                    continue

                docs_for_source = list(
                    zip(res["documents"], res["metadatas"])
                )

                docs_for_source = sorted(
                    docs_for_source,
                    key=lambda x: x[1].get("chunk_index", 0)
                )

                docs_for_source = docs_for_source[:max_chunks_per_doc]

                for text, meta in docs_for_source:

                    text_lower = text.lower()

                    # remove low quality chunks
                    if "table of contents" in text_lower or "index" in text_lower:
                        continue

                    expanded_docs.append({
                        "text": text,
                        "metadata": meta
                    })

            except Exception as e:

                logger.warning(f"Context expansion failed for {source}: {e}")

        return expanded_docs

    # -------------------------
    # LLM CALL
    # -------------------------

    def call_gemini(self, question, context, max_tokens=1000):

        if not self.api_key:
            return "Gemini API key missing."

        model = genai.GenerativeModel(self.model_name)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        retries = 3

        for attempt in range(retries):

            try:

                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens
                    )
                )

                return response.text

            except Exception as e:

                if "429" in str(e):

                    wait_time = 10
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"Gemini API error: {str(e)}")
                return f"Error calling Gemini API: {str(e)}"

        return "LLM rate limit exceeded. Please try again later."

    def call_ollama(self, question, context, max_tokens=500):

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        #print(context,question)
        try:

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2:1b",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                        "num_predict": 200,
                        "temperature": 0.2
                    }
                },
                timeout=300
            )

            data = response.json()

            return data["message"]["content"]

        except Exception as e:

            logger.error(f"Ollama error: {str(e)}")

            return f"Ollama error: {str(e)}"
    # -------------------------
    # MAIN RAG PIPELINE
    # -------------------------

    def answer(self, question, google_id, top_k, model="gemini"):

        if not question or len(question.strip()) == 0:
            return {
                "answer": "Question cannot be empty",
                "sources": [],
                "retrieved_count": 0
            }

        # -------------------------
        # RETRIEVE
        # -------------------------

        retrieved_docs = self.retrieve(
            question,
            google_id,
            k=top_k
        )

        print("retrieved_docs ",retrieved_docs)

        if not retrieved_docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "retrieved_count": 0
            }

        # -------------------------
        # SELECT SINGLE DOCUMENT
        # -------------------------

        source_counts = Counter(
            d["metadata"].get("source", "unknown")
            for d in retrieved_docs
        )

        best_source = source_counts.most_common(1)[0][0]

        docs = [
            d for d in retrieved_docs
            if d["metadata"].get("source") == best_source
        ]

        # -------------------------
        # SORT CHUNKS
        # -------------------------
        

        docs = sorted(
            docs,
            key=lambda x: x["metadata"].get("chunk_index", 0)
        )


        # -------------------------
        # BUILD CONTEXT (ONLY ONE DOC)
        # -------------------------

        context = "\n\n".join(d["text"] for d in docs)

        context = f"Document: {best_source}\n\n{context}"

        # -------------------------
        # GENERATE ANSWER
        # -------------------------
        print("Chunks sent:", len(docs))
        print("Context chars:", len(context))

        
        llm_out = self.generate(question, context, model)

        # -------------------------
        # SOURCES
        # -------------------------

        sources = [{
            "filename": best_source,
            "snippet": docs[0]["text"][:200] if docs else ""
        }]

        return {
            "answer": llm_out,
            "sources": sources,
            "retrieved_count": len(docs)
        }
    

    def generate(self, question, context, model="gemini"):


        if model == "ollama":
            return self.call_ollama(question, context)

        if model == "gemini":
            return self.call_gemini(question, context)

        return "Unknown model selected"