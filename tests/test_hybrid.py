from rag.hybrid import BM25Index, fuse_scores, tokenize


def test_tokenize_tr():
    assert "sorgulama" in tokenize("RAG Sorgulama Asistanı")


def test_bm25_ranks_relevant_doc():
    bm25 = BM25Index()
    bm25.add_many(
        [
            (1, "kedi süt içer"),
            (2, "arabalar otoyolda hızlanır"),
            (3, "kedi fare yakalar ve süt sever"),
        ]
    )
    scores = bm25.score("kedi süt")
    assert scores
    assert max(scores, key=scores.get) in (1, 3)


def test_fuse_prefers_agreement():
    dense = {1: 0.9, 2: 0.2}
    sparse = {1: 5.0, 3: 4.0}
    fused = fuse_scores(dense, sparse, alpha=0.5)
    assert 1 in fused
    assert fused[1] >= fused.get(2, 0)
