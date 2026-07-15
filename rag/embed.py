from typing import List, Optional
import numpy as np

from app.config import DEFAULT_EMBEDDING_MODEL, MULTILINGUAL_EMBEDDING_MODEL

EMBEDDING_PRESETS = {
    "mini-en": {
        "label": "MiniLM (EN, hızlı)",
        "model": DEFAULT_EMBEDDING_MODEL,
    },
    "mini-multi": {
        "label": "MiniLM Multilingual (TR uyumlu)",
        "model": MULTILINGUAL_EMBEDDING_MODEL,
    },
}


def resolve_embedding_model(preset_or_model: str) -> str:
    if preset_or_model in EMBEDDING_PRESETS:
        return EMBEDDING_PRESETS[preset_or_model]["model"]
    return preset_or_model


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = resolve_embedding_model(model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


def preset_for_model(model_name: Optional[str]) -> str:
    if not model_name:
        return "mini-en"
    for key, meta in EMBEDDING_PRESETS.items():
        if meta["model"] == model_name:
            return key
    return "mini-en"
