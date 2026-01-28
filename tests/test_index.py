import numpy as np
import pytest


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def test_faiss_index_add_and_search():
    try:
        from rag.index import FaissIndex
        from rag.types import ChunkMetadata
    except Exception:
        pytest.skip("FAISS veya ilgili bağımlılıklar mevcut değil; testi atlıyoruz.")

    dim = 3
    idx = FaissIndex(dim=dim)

    # 5 adet basit vektör (normalize)
    vecs = np.stack([
        _unit([1, 0, 0]),
        _unit([0.9, 0.1, 0]),
        _unit([0, 1, 0]),
        _unit([0, 0, 1]),
        _unit([0.5, 0.5, 0])
    ], axis=0)

    texts = [f"chunk-{i}" for i in range(vecs.shape[0])]
    metas = [
        ChunkMetadata(source_file="dummy.pdf", chunk_id=i, page_start=1, page_end=1, word_count=10)
        for i in range(vecs.shape[0])
    ]

    idx.add(vecs, texts, metas)

    q = vecs[0:1]
    res = idx.search(q, top_k=3)

    assert len(res) >= 1
    # En benzeri ilk vektör olmalı ve skor 0.95+'larda olmalı
    assert res[0].metadata.chunk_id == 0
    assert res[0].score > 0.95
