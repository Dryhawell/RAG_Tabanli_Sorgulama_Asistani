import numpy as np

from rag.index import FaissIndex
from rag.meta_store import normalize_folder, normalize_tags, upsert_source_meta
from rag.retrieve import apply_metadata_filters, retrieve
from rag.types import ChunkMetadata, RetrievedChunk


def test_normalize_helpers():
    assert normalize_folder("../hukuk/../sozlesme/") == "sozlesme"
    assert normalize_tags("A, b; A") == ["A", "b"]


def test_apply_folder_and_tag_filters():
    chunks = [
        RetrievedChunk(
            chunk_id=0,
            score=0.9,
            text="a",
            metadata=ChunkMetadata(
                source_file="hukuk/a.txt",
                chunk_id=0,
                page_start=1,
                page_end=1,
                word_count=1,
                folder="hukuk",
                tags=["sozlesme", "2024"],
            ),
        ),
        RetrievedChunk(
            chunk_id=1,
            score=0.8,
            text="b",
            metadata=ChunkMetadata(
                source_file="genel.txt",
                chunk_id=1,
                page_start=1,
                page_end=1,
                word_count=1,
                folder="",
                tags=["not"],
            ),
        ),
    ]
    only_hukuk = apply_metadata_filters(chunks, folder_filter=["hukuk"])
    assert [c.chunk_id for c in only_hukuk] == [0]

    tagged = apply_metadata_filters(chunks, tag_filter=["sozlesme"])
    assert [c.chunk_id for c in tagged] == [0]

    all_mode = apply_metadata_filters(
        chunks, tag_filter=["sozlesme", "eksik"], tag_mode="all"
    )
    assert all_mode == []


def test_retrieve_respects_tag_filter():
    idx = FaissIndex(dim=2)
    vecs = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    texts = ["kedi süt", "kedi mama"]
    metas = [
        ChunkMetadata(
            source_file="a.txt",
            chunk_id=0,
            page_start=1,
            page_end=1,
            word_count=2,
            folder="hayvan",
            tags=["evcil"],
        ),
        ChunkMetadata(
            source_file="b.txt",
            chunk_id=1,
            page_start=1,
            page_end=1,
            word_count=2,
            folder="diger",
            tags=["yemek"],
        ),
    ]
    idx.add(vecs, texts, metas)
    hits, gate = retrieve(
        idx,
        vecs[0:1],
        "kedi",
        use_hybrid=False,
        top_k=2,
        tag_filter=["evcil"],
        threshold=0.1,
    )
    assert hits
    assert all("evcil" in h.metadata.tags for h in hits)
    assert gate >= 0.1


def test_upsert_source_meta(tmp_path):
    path = str(tmp_path / "sources.json")
    entry = upsert_source_meta(
        "hukuk/a.pdf", folder="hukuk", tags="x,y", path=path
    )
    assert entry["folder"] == "hukuk"
    assert entry["tags"] == ["x", "y"]
