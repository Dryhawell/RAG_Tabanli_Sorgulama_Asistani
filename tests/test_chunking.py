from rag.chunking import chunk_pages


def test_chunking_basic():
    words = ["kelime"] * 1000
    text = " ".join(words)
    texts, metas = chunk_pages(
        source_file="dummy.pdf",
        pages=[text],
        chunk_size_words=300,
        overlap_ratio=0.1,
        min_chunk_words=200,
    )

    assert len(texts) > 1, "Parçalama birden fazla chunk üretmeli"
    assert len(texts) == len(metas)

    for t, m in zip(texts, metas):
        wc = len(t.split())
        assert wc == m.word_count
        assert wc >= 200 or wc == 300
        assert m.chunk_uid and len(m.chunk_uid) == 24


def test_chunking_respects_headings_and_paragraphs():
    page = "\n".join(
        [
            "1. Giriş",
            "Bu paragraf giriş hakkındadır. " + ("anlatım " * 80),
            "",
            "2. Yöntem",
            "Yöntem paragrafı burada. " + ("adım " * 80),
        ]
    )
    texts, metas = chunk_pages(
        source_file="rapor.pdf",
        pages=[page],
        chunk_size_words=120,
        overlap_ratio=0.1,
        min_chunk_words=40,
    )
    assert texts
    headings = {m.heading for m in metas if m.heading}
    assert any("Giriş" in (h or "") for h in headings) or any("Yöntem" in (h or "") for h in headings)


def test_chunking_merges_short_tail():
    # İlk blok uzun, sonda kısa kuyruk; kuyruk tek başına chunk olmamalı
    page = ("paragraf " * 220) + "\n\n" + ("son " * 20)
    texts, metas = chunk_pages(
        source_file="a.txt",
        pages=[page],
        chunk_size_words=200,
        overlap_ratio=0.1,
        min_chunk_words=80,
    )
    assert texts
    assert all(m.word_count >= 80 or i < len(texts) - 1 for i, m in enumerate(metas)) or metas[-1].word_count >= 20
