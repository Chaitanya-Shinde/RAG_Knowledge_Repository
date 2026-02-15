# app/db_client.py
import chromadb
from chromadb.utils import embedding_functions
import os

class ChromaClient:
    def __init__(self, client_settings=None):
        # """
        # New-style chromadb client wrapper.
        # If client_settings contains 'persist_directory', we'll use PersistentClient(path=...)
        # Otherwise we create a regular in-memory client via chromadb.Client().
        # """
        # persist_dir = None
        # if client_settings and 'persist_directory' in client_settings:
        #     persist_dir = client_settings['persist_directory']

        # if persist_dir:
        #     # ensure directory exists
        #     os.makedirs(persist_dir, exist_ok=True)
        #     try:
        #         # PersistentClient is the new API for disk persistence
        #         self.client = chromadb.PersistentClient(path=persist_dir)
        #     except Exception as e:
        #         # fallback to plain Client to surface helpful error
        #         raise RuntimeError(f"Failed to create PersistentClient at {persist_dir}: {e}")
        # else:
        #     # in-memory client
        #     self.client = chromadb.Client()

        # # Use a single collection instance; create if not exists
        # self.collection = None

        
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
