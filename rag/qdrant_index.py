"""Qdrant tabanlı vektör indeksi (FAISS ile aynı arayüz).

Uzak sunucu (`RAG_QDRANT_URL`) veya yerel gömülü (`RAG_QDRANT_PATH` / :memory:).
Payload içinde text + metadata tutulur; BM25 için `_id_to_text` / `_id_to_meta` senkron kalır.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

import numpy as np

from .types import ChunkMetadata, RetrievedChunk


def _require_qdrant():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest
    except Exception as exc:
        raise RuntimeError(
            "qdrant-client kurulu değil. pip install qdrant-client"
        ) from exc
    return QdrantClient, rest


class QdrantIndex:
    def __init__(
        self,
        dim: int,
        *,
        embedding_model: str | None = None,
        collection: str = "rag_chunks",
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        client=None,
    ):
        QdrantClient, rest = _require_qdrant()
        self.dim = dim
        self.embedding_model = embedding_model
        self.collection = collection or "rag_chunks"
        self._rest = rest
        self._id_to_meta: Dict[int, ChunkMetadata] = {}
        self._id_to_text: Dict[int, str] = {}
        self._next_id = 0

        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
        elif path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        rest = self._rest
        try:
            exists = self.client.collection_exists(self.collection)
        except Exception:
            names = {c.name for c in self.client.get_collections().collections}
            exists = self.collection in names
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(
                    size=self.dim,
                    distance=rest.Distance.COSINE,
                ),
            )

    @property
    def size(self) -> int:
        return len(self._id_to_text)

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[ChunkMetadata]):
        assert embeddings.shape[0] == len(texts) == len(metas)
        if embeddings.shape[0] == 0:
            return
        ids = list(range(self._next_id, self._next_id + embeddings.shape[0]))
        points = []
        for i, cid in enumerate(ids):
            meta = metas[i]
            payload = {
                "text": texts[i],
                "meta": meta.model_dump(),
                "source_file": meta.source_file,
                "folder": meta.folder or "",
                "tags": list(meta.tags or []),
                "chunk_uid": str(getattr(meta, "chunk_uid", None) or ""),
            }
            points.append(
                self._rest.PointStruct(
                    id=int(cid),
                    vector=embeddings[i].astype(np.float32).tolist(),
                    payload=payload,
                )
            )
            self._id_to_meta[int(cid)] = meta
            self._id_to_text[int(cid)] = texts[i]
        self.client.upsert(collection_name=self.collection, points=points)
        self._next_id += embeddings.shape[0]

    def _scroll_point_ids(self, query_filter) -> List[int]:
        """Qdrant payload filter ile point id listesi."""
        out: List[int] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for rec in records:
                out.append(int(rec.id))
            if offset is None:
                break
        return out

    def ids_for_source(self, source_file: str, *, use_payload: bool = True) -> List[int]:
        mem = [
            cid
            for cid, meta in self._id_to_meta.items()
            if meta.source_file == source_file
        ]
        if not use_payload:
            return mem
        try:
            flt = self._rest.Filter(
                must=[
                    self._rest.FieldCondition(
                        key="source_file",
                        match=self._rest.MatchValue(value=str(source_file)),
                    )
                ]
            )
            remote = self._scroll_point_ids(flt)
            if remote:
                return sorted(set(remote))
        except Exception:
            pass
        return mem

    def ids_for_chunk_uids(
        self,
        source_file: str,
        uids: Set[str] | List[str],
        *,
        use_payload: bool = True,
    ) -> List[int]:
        want = {str(u) for u in uids if u}
        if not want:
            return []
        mem: List[int] = []
        for cid, meta in self._id_to_meta.items():
            if meta.source_file != source_file:
                continue
            uid = getattr(meta, "chunk_uid", None)
            if uid and str(uid) in want:
                mem.append(cid)
        if not use_payload:
            return mem
        try:
            flt = self._rest.Filter(
                must=[
                    self._rest.FieldCondition(
                        key="source_file",
                        match=self._rest.MatchValue(value=str(source_file)),
                    ),
                    self._rest.FieldCondition(
                        key="chunk_uid",
                        match=self._rest.MatchAny(any=sorted(want)),
                    ),
                ]
            )
            remote = self._scroll_point_ids(flt)
            if remote:
                return sorted(set(remote))
        except Exception:
            pass
        return mem

    def list_sources(self) -> List[str]:
        sources: Set[str] = {meta.source_file for meta in self._id_to_meta.values()}
        return sorted(sources)

    def list_folders(self) -> List[str]:
        folders: Set[str] = {
            (meta.folder or "").strip()
            for meta in self._id_to_meta.values()
            if (meta.folder or "").strip()
        }
        return sorted(folders)

    def list_tags(self) -> List[str]:
        tags: Set[str] = set()
        for meta in self._id_to_meta.values():
            for t in meta.tags or []:
                if t and str(t).strip():
                    tags.add(str(t).strip())
        return sorted(tags, key=lambda x: x.lower())

    def remove_ids(self, ids: List[int]) -> int:
        if not ids:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=self._rest.PointIdsList(points=[int(i) for i in ids]),
        )
        removed = 0
        for cid in ids:
            if cid in self._id_to_text:
                removed += 1
            self._id_to_meta.pop(cid, None)
            self._id_to_text.pop(cid, None)
        return removed

    def remove_source(self, source_file: str) -> int:
        return self.remove_ids(self.ids_for_source(source_file))

    def replace_source(
        self,
        source_file: str,
        embeddings: np.ndarray,
        texts: List[str],
        metas: List[ChunkMetadata],
    ) -> int:
        removed = self.remove_source(source_file)
        self.add(embeddings, texts, metas)
        return removed

    def search(self, query: np.ndarray, top_k: int = 6) -> List[RetrievedChunk]:
        if self.size == 0:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        limit = min(top_k, self.size)
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query[0].astype(np.float32).tolist(),
            limit=limit,
            with_payload=True,
        )
        hits = getattr(resp, "points", None) or resp
        out: List[RetrievedChunk] = []
        for hit in hits:
            cid = int(hit.id)
            payload = hit.payload or {}
            text = payload.get("text") or self._id_to_text.get(cid)
            meta_raw = payload.get("meta")
            if meta_raw and isinstance(meta_raw, dict):
                meta = ChunkMetadata(**meta_raw)
            else:
                meta = self._id_to_meta.get(cid)
            if meta is None or text is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    score=float(hit.score),
                    text=text,
                    metadata=meta,
                )
            )
        return out

    def _reload_from_client(self) -> None:
        """Collection'daki tüm noktaları belleğe çeker."""
        self._id_to_meta.clear()
        self._id_to_text.clear()
        offset = None
        max_id = -1
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                cid = int(rec.id)
                payload = rec.payload or {}
                text = payload.get("text") or ""
                meta_raw = payload.get("meta") or {}
                if isinstance(meta_raw, dict) and meta_raw.get("source_file"):
                    meta = ChunkMetadata(**meta_raw)
                    if not getattr(meta, "chunk_uid", None) and payload.get("chunk_uid"):
                        meta.chunk_uid = str(payload.get("chunk_uid") or "") or None
                else:
                    meta = ChunkMetadata(
                        source_file=str(payload.get("source_file") or "unknown"),
                        chunk_id=cid,
                        page_start=0,
                        page_end=0,
                        word_count=len(text.split()),
                        folder=str(payload.get("folder") or ""),
                        tags=list(payload.get("tags") or []),
                        chunk_uid=str(payload.get("chunk_uid") or "") or None,
                    )
                self._id_to_meta[cid] = meta
                self._id_to_text[cid] = text
                max_id = max(max_id, cid)
            if offset is None:
                break
        if self._next_id <= max_id:
            self._next_id = max_id + 1

    def save(self, index_path: str, docstore_path: str):
        """Sidecar meta yazar; vektörler Qdrant collection'da kalır."""
        os.makedirs(os.path.dirname(docstore_path) or ".", exist_ok=True)
        # FAISS path yerine backend işaretçisi (isteğe bağlı)
        if index_path:
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backend": "qdrant",
                        "collection": self.collection,
                        "dim": self.dim,
                    },
                    f,
                )
        data = {
            "backend": "qdrant",
            "dim": self.dim,
            "next_id": self._next_id,
            "embedding_model": self.embedding_model,
            "collection": self.collection,
            "items": {
                str(k): {
                    "text": self._id_to_text[k],
                    "meta": self._id_to_meta[k].model_dump(),
                }
                for k in self._id_to_text
            },
        }
        with open(docstore_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        index_path: str,
        docstore_path: str,
        *,
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        client=None,
    ):
        with open(docstore_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        coll = collection or data.get("collection") or "rag_chunks"
        fi = cls(
            dim=int(data.get("dim") or 384),
            embedding_model=data.get("embedding_model"),
            collection=coll,
            url=url,
            path=path,
            api_key=api_key,
            client=client,
        )
        # Önce Qdrant'tan, yoksa sidecar items
        try:
            count = fi.client.count(collection_name=fi.collection, exact=True).count
        except Exception:
            count = 0
        if count > 0:
            fi._next_id = int(data.get("next_id") or 0)
            fi._reload_from_client()
        else:
            items = data.get("items") or {}
            # Yeniden eklemek için vektör yok → yalnızca memory map (arama boş)
            for k, v in items.items():
                meta = ChunkMetadata(**v["meta"]) if isinstance(v["meta"], dict) else ChunkMetadata.model_validate(v["meta"])
                fi._id_to_meta[int(k)] = meta
                fi._id_to_text[int(k)] = v["text"]
            if "next_id" in data:
                fi._next_id = int(data["next_id"])
            elif fi._id_to_text:
                fi._next_id = max(fi._id_to_text.keys()) + 1
        return fi
