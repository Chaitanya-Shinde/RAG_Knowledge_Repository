import os
import logging
from .embeddings import EmbeddingModel
from .db_client import ChromaClient
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = '''You are a helpful assistant. Use ONLY the context given below to answer the question. Provide concise answer and list sources at the end.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
'''

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

    def retrieve(self, query, k=5):
        q_emb = self.embed_model.embed_query(query)
        res = self.db.query(q_emb, n=k)
        docs = []
        if res and 'documents' in res:
            for doc_list, meta_list in zip(res['documents'], res['metadatas']):
                for d, m in zip(doc_list, meta_list):
                    docs.append({"text": d, "metadata": m})
        logger.debug(f"Retrieved {len(docs)} documents for query")
        return docs

    def call_llm(self, prompt, max_tokens=512):
        if not self.api_key:
            return "Gemini API key missing. Set GEMINI_API_KEY in .env."

        try:
            model = genai.GenerativeModel(self.model_name)
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

    def answer(self, question, top_k=5):
        if not question or len(question.strip()) == 0:
            return {"answer": "Question cannot be empty", "sources": [], "retrieved_count": 0}
        
        docs = self.retrieve(question, k=top_k)
        context = "\n\n".join([f"Source: {d['metadata'].get('source','unknown')}\n"+d['text'] for d in docs])
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        llm_out = self.call_llm(prompt)
        sources = list({d['metadata'].get('source','unknown') for d in docs})
        
        return {"answer": llm_out, "sources": sources, "retrieved_count": len(docs)}
