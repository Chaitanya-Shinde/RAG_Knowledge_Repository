# app/db_client.py
import chromadb
from chromadb.utils import embedding_functions
import os

class ChromaClient:
    def __init__(self, client_settings=None):
         
        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE", "RAKR")

        if api_key and tenant:
            # Chroma Cloud
            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database
            )
        else:
            raise RuntimeError("Chroma Cloud credentials missing")

        self.collection = None


    def ensure_collection(self, name="ragr"):
        if self.collection is None:
            # create_collection works with the new client
            try:
                # create_collection raises if exists; use get_collection if you prefer
                self.collection = self.client.get_or_create_collection(name=name)
            except Exception:
                self.collection = self.client.create_collection(name=name)
        return self.collection

    def add_documents(self, ids, texts, metadatas, embeddings):
        col = self.ensure_collection()
        col.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings)
        

    def query(self, query_embedding, n=5):
        col = self.ensure_collection()
        # request top-n results using the new API
        results = col.query(query_embeddings=[query_embedding], n_results=n)
        return results

    def get_user_collection(self, google_id: str):
        return self.client.get_or_create_collection(
            name=f"user_{google_id}"
        )