from rag.chunking import chunk_pages


def test_chunking_basic():
    # 1000 kelimelik bir sayfa oluştur
    words = ["kelime"] * 1000
    text = " ".join(words)
    texts, metas = chunk_pages(source_file="dummy.pdf", pages=[text], chunk_size_words=300, overlap_ratio=0.1)

    assert len(texts) > 1, "Parçalama birden fazla chunk üretmeli"
    assert len(texts) == len(metas)

    # Her meta word_count ile metin uyumlu olmalı
    for t, m in zip(texts, metas):
        wc = len(t.split())
        assert wc == m.word_count
        # Çok küçük son parça üretilmemeli (heuristic)
        assert wc >= 200 or wc == 300
