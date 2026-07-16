"""Model indirmeden çalışan deterministik hash embedding (eval/regression)."""

from __future__ import annotations

from typing import List

import numpy as np

from rag.hybrid import tokenize


class HashEmbedder:
    """Token hash'lerini sabit boyutta vektöre yığar; L2 normalize eder."""

    def __init__(self, dim: int = 64, model_name: str = "hash-embedder"):
        self.dim = dim
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                idx = hash(tok) % self.dim
                out[i, idx] += 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out
