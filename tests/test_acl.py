from rag.acl import (
    can_access_meta,
    can_ingest_to,
    filter_chunks,
    intersect_folder_filter,
    intersect_tag_filter,
)
from rag.auth import User
from rag.types import ChunkMetadata, RetrievedChunk


def _chunk(folder: str, tags: list, cid: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        score=0.5,
        text="x",
        metadata=ChunkMetadata(
            source_file="a.txt",
            chunk_id=cid,
            page_start=1,
            page_end=1,
            word_count=1,
            folder=folder,
            tags=tags,
        ),
    )


def test_admin_sees_all():
    admin = User("admin", role="admin")
    assert can_access_meta(admin, folder="gizli", tags=["secret"])


def test_user_folder_acl():
    user = User("demo", role="user", allowed_folders=["hukuk"], allowed_tags=None)
    assert can_access_meta(user, folder="hukuk", tags=["x"])
    assert not can_access_meta(user, folder="ik", tags=["x"])
    assert intersect_folder_filter(user, None) == ["hukuk"]
    assert intersect_folder_filter(user, ["hukuk", "ik"]) == ["hukuk"]


def test_user_tag_acl():
    user = User("demo", role="user", allowed_folders=None, allowed_tags=["public"])
    assert can_access_meta(user, folder="", tags=["public"])
    assert not can_access_meta(user, folder="", tags=["internal"])
    assert can_access_meta(user, folder="", tags=[])  # etiketsiz serbest
    assert intersect_tag_filter(user, ["public", "internal"]) == ["public"]


def test_filter_chunks_and_ingest(monkeypatch):
    monkeypatch.setattr("rag.auth.AUTH_USER_CAN_INGEST", True)
    user = User("demo", role="user", allowed_folders=["hukuk"], allowed_tags=["public"])
    chunks = [
        _chunk("hukuk", ["public"], 0),
        _chunk("ik", ["public"], 1),
        _chunk("hukuk", ["secret"], 2),
    ]
    kept = filter_chunks(user, chunks)
    assert [c.chunk_id for c in kept] == [0]

    assert can_ingest_to(user, folder="hukuk", tags=["public"])
    assert not can_ingest_to(user, folder="ik", tags=["public"])
