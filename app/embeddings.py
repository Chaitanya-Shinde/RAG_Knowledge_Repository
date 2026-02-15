from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def embed_query(self, text):
        arr = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return arr[0]
