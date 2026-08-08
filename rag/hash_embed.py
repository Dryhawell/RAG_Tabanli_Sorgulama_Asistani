"""Model indirmeden çalışan deterministik hash embedding (eval/regression)."""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from rag.hybrid import tokenize


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


class HashEmbedder:
    """Token hash'lerini sabit boyutta vektöre yığar; L2 normalize eder.

    Python'un rastgele `hash()`'i yerine MD5 kullanır; süreçler arası kararlıdır.
    """

    def __init__(self, dim: int = 64, model_name: str = "hash-embedder"):
        self.dim = dim
        self.model_name = model_name

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                idx = _stable_bucket(tok, self.dim)
                out[i, idx] += 1.0
            n = float(np.linalg.norm(out[i]))
            if n > 0:
                out[i] /= n
        return out
