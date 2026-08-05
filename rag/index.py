import json
from typing import Dict, List, Set
import os
import numpy as np
import faiss
from .types import ChunkMetadata, RetrievedChunk


class FaissIndex:
    def __init__(self, dim: int, embedding_model: str | None = None):
        self.dim = dim
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatIP(dim)
        self.idmap = faiss.IndexIDMap2(self.index)
        self._id_to_meta: Dict[int, ChunkMetadata] = {}
        self._id_to_text: Dict[int, str] = {}
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(getattr(self.idmap, "ntotal", 0))

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[ChunkMetadata]):
        assert embeddings.shape[0] == len(texts) == len(metas)
        if embeddings.shape[0] == 0:
            return
        ids = np.arange(self._next_id, self._next_id + embeddings.shape[0], dtype=np.int64)
        self.idmap.add_with_ids(embeddings, ids)
        for i, idv in enumerate(ids):
            self._id_to_meta[int(idv)] = metas[i]
            self._id_to_text[int(idv)] = texts[i]
        self._next_id += embeddings.shape[0]

    def ids_for_source(self, source_file: str) -> List[int]:
        return [
            cid
            for cid, meta in self._id_to_meta.items()
            if meta.source_file == source_file
        ]

    def ids_for_chunk_uids(self, source_file: str, uids: Set[str] | List[str]) -> List[int]:
        want = {str(u) for u in uids if u}
        if not want:
            return []
        out: List[int] = []
        for cid, meta in self._id_to_meta.items():
            if meta.source_file != source_file:
                continue
            uid = getattr(meta, "chunk_uid", None)
            if uid and str(uid) in want:
                out.append(cid)
        return out

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
        selector = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
        removed = int(self.idmap.remove_ids(selector))
        for cid in ids:
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
        """Aynı kaynak varsa chunk'larını silip yenilerini ekler. Dönüş: silinen chunk sayısı."""
        removed = self.remove_source(source_file)
        self.add(embeddings, texts, metas)
        return removed

    def search(self, query: np.ndarray, top_k: int = 6) -> List[RetrievedChunk]:
        if self.size == 0:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, ids = self.idmap.search(query, min(top_k, self.size))
        out: List[RetrievedChunk] = []
        for i in range(ids.shape[1]):
            cid = int(ids[0, i])
            if cid == -1:
                continue
            score = float(scores[0, i])
            meta = self._id_to_meta.get(cid)
            text = self._id_to_text.get(cid)
            if meta is None or text is None:
                continue
            out.append(RetrievedChunk(chunk_id=cid, score=score, text=text, metadata=meta))
        return out

    def save(self, index_path: str, docstore_path: str):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(docstore_path) or ".", exist_ok=True)
        faiss.write_index(self.idmap, index_path)
        data = {
            "dim": self.dim,
            "next_id": self._next_id,
            "embedding_model": self.embedding_model,
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

    @staticmethod
    def load(index_path: str, docstore_path: str):
        idmap = faiss.read_index(index_path)
        with open(docstore_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fi = FaissIndex(
            dim=data.get("dim", idmap.d),
            embedding_model=data.get("embedding_model"),
        )
        fi.idmap = idmap
        items = data.get("items", {})
        for k, v in items.items():
            meta = (
                ChunkMetadata(**v["meta"])
                if isinstance(v["meta"], dict)
                else ChunkMetadata.model_validate(v["meta"])
            )
            fi._id_to_meta[int(k)] = meta
            fi._id_to_text[int(k)] = v["text"]
        if "next_id" in data:
            fi._next_id = int(data["next_id"])
        elif fi._id_to_text:
            fi._next_id = max(fi._id_to_text.keys()) + 1
        else:
            fi._next_id = 0
        return fi
