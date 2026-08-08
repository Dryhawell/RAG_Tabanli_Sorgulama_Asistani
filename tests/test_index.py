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

    vecs = np.stack(
        [
            _unit([1, 0, 0]),
            _unit([0.9, 0.1, 0]),
            _unit([0, 1, 0]),
            _unit([0, 0, 1]),
            _unit([0.5, 0.5, 0]),
        ],
        axis=0,
    )

    texts = [f"chunk-{i}" for i in range(vecs.shape[0])]
    metas = [
        ChunkMetadata(source_file="dummy.pdf", chunk_id=i, page_start=1, page_end=1, word_count=10)
        for i in range(vecs.shape[0])
    ]

    idx.add(vecs, texts, metas)

    q = vecs[0:1]
    res = idx.search(q, top_k=3)

    assert len(res) >= 1
    assert res[0].metadata.chunk_id == 0
    assert res[0].score > 0.95


def test_faiss_replace_source_dedups():
    try:
        from rag.index import FaissIndex
        from rag.types import ChunkMetadata
    except Exception:
        pytest.skip("FAISS veya ilgili bağımlılıklar mevcut değil; testi atlıyoruz.")

    idx = FaissIndex(dim=2)
    v1 = np.stack([_unit([1, 0]), _unit([0, 1])], axis=0)
    metas1 = [
        ChunkMetadata(source_file="a.txt", chunk_id=0, page_start=1, page_end=1, word_count=3),
        ChunkMetadata(source_file="a.txt", chunk_id=1, page_start=1, page_end=1, word_count=3),
    ]
    idx.add(v1, ["eski-1", "eski-2"], metas1)
    assert idx.size == 2

    v2 = np.stack([_unit([0.2, 0.8])], axis=0)
    metas2 = [
        ChunkMetadata(source_file="a.txt", chunk_id=0, page_start=1, page_end=1, word_count=2),
    ]
    removed = idx.replace_source("a.txt", v2, ["yeni"], metas2)
    assert removed == 2
    assert idx.size == 1
    assert idx.list_sources() == ["a.txt"]
    assert idx.search(v2, top_k=1)[0].text == "yeni"
