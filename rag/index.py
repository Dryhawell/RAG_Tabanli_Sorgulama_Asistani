import json
from typing import List, Tuple, Dict
import os
import numpy as np
import faiss
from .types import ChunkMetadata, RetrievedChunk

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.idmap = faiss.IndexIDMap2(self.index)
        self._id_to_meta: Dict[int, ChunkMetadata] = {}
        self._id_to_text: Dict[int, str] = {}
        self._next_id = 0

    def add(self, embeddings: np.ndarray, texts: List[str], metas: List[ChunkMetadata]):
        assert embeddings.shape[0] == len(texts) == len(metas)
        ids = np.arange(self._next_id, self._next_id + embeddings.shape[0], dtype=np.int64)
        self.idmap.add_with_ids(embeddings, ids)
        for i, idv in enumerate(ids):
            self._id_to_meta[int(idv)] = metas[i]
            self._id_to_text[int(idv)] = texts[i]
        self._next_id += embeddings.shape[0]

    def search(self, query: np.ndarray, top_k: int = 6) -> List[RetrievedChunk]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, ids = self.idmap.search(query, top_k)
        out: List[RetrievedChunk] = []
        for i in range(ids.shape[1]):
            cid = int(ids[0, i])
            if cid == -1:
                continue
            score = float(scores[0, i])
            meta = self._id_to_meta[cid]
            text = self._id_to_text[cid]
            out.append(RetrievedChunk(chunk_id=cid, score=score, text=text, metadata=meta))
        return out

    def save(self, index_path: str, docstore_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(docstore_path), exist_ok=True)
        faiss.write_index(self.idmap, index_path)
        # Docstore
        data = {
            "dim": self.dim,
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
        fi = FaissIndex(dim=data.get("dim", idmap.d))
        fi.idmap = idmap
        items = data.get("items", {})
        for k, v in items.items():
            meta = ChunkMetadata(**v["meta"]) if isinstance(v["meta"], dict) else ChunkMetadata.parse_obj(v["meta"]) 
            fi._id_to_meta[int(k)] = meta
            fi._id_to_text[int(k)] = v["text"]
        fi._next_id = len(fi._id_to_text)
        return fi
