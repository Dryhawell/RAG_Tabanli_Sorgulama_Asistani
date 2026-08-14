import numpy as np

from rag.compare import collect_source_bundles, compare_sources, summarize_sources
from rag.hash_embed import HashEmbedder
from rag.index import FaissIndex
from rag.query_rewrite import embed_rewrite, expand_query, generate_hyde_passage, rewrite_query
from rag.types import ChunkMetadata


def test_hyde_and_expand_stubs():
    def fake_hyde(prompt: str) -> str:
        assert "Soru:" in prompt
        return "Yıllık izin 14 gündür ve önceden onay gerekir.\n\nBaşka satır"

    def fake_expand(prompt: str) -> str:
        return "izin süresi nedir\nkaç gün yıllık izin var"

    hypo = generate_hyde_passage("izin kaç gün?", fake_hyde)
    assert "14 gün" in hypo
    assert "Başka satır" not in hypo

    alts = expand_query("izin kaç gün?", fake_expand)
    assert len(alts) == 2

    r = rewrite_query("izin kaç gün?", mode="hyde", generate_fn=fake_hyde)
    assert r.hypothetical and "14" in r.hypothetical
    assert r.retrieval_texts[0] == r.hypothetical

    r2 = rewrite_query("izin kaç gün?", mode="expand", generate_fn=fake_expand)
    assert len(r2.expansions) == 2
    assert "izin" in r2.bm25_query.lower()

    r3 = rewrite_query("izin kaç gün?", mode="hyde+expand", generate_fn=lambda p: fake_hyde(p) if "Paragraf" in p else fake_expand(p))
    assert r3.hypothetical
    assert r3.expansions


def test_embed_rewrite_mean():
    emb = HashEmbedder(dim=32)
    r = rewrite_query("soru", mode="none")
    v = embed_rewrite(emb, r)
    assert v.shape == (1, 32)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def _build_tiny_index():
    idx = FaissIndex(dim=4)
    vecs = np.eye(3, 4, dtype=np.float32)
    metas = [
        ChunkMetadata(source_file="a.txt", chunk_id=0, page_start=1, page_end=1, word_count=5),
        ChunkMetadata(source_file="a.txt", chunk_id=1, page_start=1, page_end=1, word_count=5),
        ChunkMetadata(source_file="b.txt", chunk_id=0, page_start=1, page_end=1, word_count=5),
    ]
    texts = [
        "Politika A: yıllık izin 14 gün.",
        "Politika A: hastalık izni ayrı düzenlenir.",
        "Politika B: yıllık izin 20 gün.",
    ]
    idx.add(vecs, texts, metas)
    return idx


def test_compare_and_summarize():
    idx = _build_tiny_index()
    bundles = collect_source_bundles(idx, ["a.txt", "b.txt"], max_chunks_per=2)
    assert len(bundles) == 2
    assert bundles[0].source_file == "a.txt"
    assert len(bundles[0].chunks) == 2

    summaries = summarize_sources(
        idx,
        ["a.txt"],
        generate_fn=lambda p: "Özet: 14 gün izin",
    )
    assert summaries["a.txt"].startswith("Özet")

    cmp = compare_sources(
        idx,
        ["a.txt", "b.txt"],
        generate_fn=lambda p: "A 14 gün, B 20 gün",
        focus="izin süreleri",
    )
    assert "14" in cmp
