import numpy as np

from rag.index import FaissIndex
from rag.rerank import LexicalReranker
from rag.retrieve import retrieve
from rag.types import ChunkMetadata


def test_retrieve_with_lexical_rerank():
    idx = FaissIndex(dim=2, embedding_model="fake")
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    texts = ["kedi süt içer", "araba hızlanır"]
    metas = [
        ChunkMetadata(source_file="hayvan.txt", chunk_id=0, page_start=1, page_end=1, word_count=3),
        ChunkMetadata(source_file="trafik.txt", chunk_id=1, page_start=1, page_end=1, word_count=2),
    ]
    idx.add(vecs, texts, metas)

    q = vecs[0:1]
    hits, gate = retrieve(
        idx,
        q,
        "kedi süt",
        bm25=None,
        top_k=1,
        use_hybrid=False,
        use_reranker=True,
        reranker=LexicalReranker(),
        threshold=0.1,
    )
    assert hits
    assert hits[0].metadata.source_file == "hayvan.txt"
    assert gate >= 0.1
