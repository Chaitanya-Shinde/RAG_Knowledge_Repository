import os
import logging
from .embeddings import EmbeddingModel
from .db_client import ChromaClient
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant.

Use ONLY the context below to answer the question.

If the context contains questions, list them exactly as written.

Do NOT summarize or invent information.

CONTEXT:
{context}

QUESTION:
{question}

Return the answer as a numbered list when possible.

ANSWER:
"""

class RAGSystem:
    def __init__(self, embed_model: EmbeddingModel, db_client: ChromaClient):
        self.embed_model = embed_model
        self.db = db_client
        self.db.ensure_collection()
        
        # Configure Gemini once
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def index_documents(self, docs):
        ids = [d['id'] for d in docs]
        texts = [d['text'] for d in docs]
        metas = [d.get('metadata',{}) for d in docs]
        embeddings = self.embed_model.embed_documents(texts)
        self.db.add_documents(ids, texts, metas, embeddings.tolist())
        logger.info(f"Indexed {len(docs)} documents")
        return {"indexed": len(docs)}

    def retrieve(self, query, google_id, k=5):
        q_emb = self.embed_model.embed_query(query)

        collection = self.db.get_user_collection(google_id)

        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )

        docs = []

        if res and "documents" in res:
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                docs.append({
                    "text": doc,
                    "metadata": meta
                })
        print("Retrieved docs:", len(docs))
        
        return docs

    def call_llm(self, question, context, max_tokens=512):

        if not self.api_key:
            return "Gemini API key missing. Set GEMINI_API_KEY in .env."

        try:
            max_tokens = int(max_tokens)

            model = genai.GenerativeModel(self.model_name)

            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens
                )
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return f"Error calling Gemini API: {str(e)}"

    def answer(self, question, google_id, top_k):
        if not question or len(question.strip()) == 0:
            return {"answer": "Question cannot be empty", "sources": [], "retrieved_count": 0}
        
        docs = self.retrieve(question, google_id, k=top_k)

        context = "\n\n".join([
            f"[Source: {d['metadata'].get('source','unknown')}]\n{d['text']}"
            for d in docs
        ])

        llm_out = self.call_llm(question, context)

        sources = list({
            d['metadata'].get('source', 'unknown')
            for d in docs
        })

        return {
            "answer": llm_out,
            "sources": sources,
            "retrieved_count": len(docs)
        }