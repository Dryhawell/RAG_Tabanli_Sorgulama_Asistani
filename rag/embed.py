from typing import List
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # normalize_embeddings=True zaten L2 normalize eder; ek olarak güvenlik için float32
        return vecs.astype(np.float32)
