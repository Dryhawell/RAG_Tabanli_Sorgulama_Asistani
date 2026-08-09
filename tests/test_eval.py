import numpy as np

from rag.eval import EvalCase, evaluate_cases, summarize
from rag.index import FaissIndex
from rag.types import ChunkMetadata


class FakeEmbedder:
    dim = 2

    def encode(self, texts):
        # "kedi" sorusu ilk vektöre yakın
        out = []
        for t in texts:
            if "kedi" in t.lower():
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        arr = np.asarray(out, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr


def _index():
    idx = FaissIndex(dim=2, embedding_model="fake")
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    texts = ["kedi süt içer", "araba hızlanır"]
    metas = [
        ChunkMetadata(source_file="hayvan.txt", chunk_id=0, page_start=1, page_end=1, word_count=3),
        ChunkMetadata(source_file="trafik.txt", chunk_id=1, page_start=1, page_end=1, word_count=2),
    ]
    idx.add(vecs, texts, metas)
    return idx


def test_evaluate_hit_and_no_answer():
    idx = _index()
    emb = FakeEmbedder()
    results = evaluate_cases(
        idx,
        emb,
        [
            EvalCase(question="kedi ne içer?", expected_source="hayvan.txt"),
            EvalCase(question="uzaylılar nerede?", expect_no_answer=True),
        ],
        use_hybrid=True,
        threshold=0.2,
    )
    summary = summarize(results)
    assert summary["total"] == 2
    assert results[0].passed
    assert summary["passed"] >= 1
