"""Kaynak bazlı klasör/etiket sidecar deposu."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from app.config import METADATA_DIR

SOURCE_META_PATH = os.path.join(METADATA_DIR, "sources.json")


def _default_path() -> str:
    return SOURCE_META_PATH


def load_source_meta(path: str | None = None) -> Dict[str, dict]:
    p = path or _default_path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_source_meta(meta: Dict[str, dict], path: str | None = None) -> None:
    p = path or _default_path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def normalize_tags(tags: Optional[List[str] | str]) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = [t.strip() for t in tags.replace(";", ",").split(",")]
    else:
        parts = [str(t).strip() for t in tags]
    out = []
    seen = set()
    for t in parts:
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def normalize_folder(folder: Optional[str]) -> str:
    if not folder:
        return ""
    f = folder.replace("\\", "/").strip().strip("/")
    # path traversal engeli + ".." çözümü
    parts: List[str] = []
    for p in f.split("/"):
        if not p or p == ".":
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/".join(parts)


def upsert_source_meta(
    source_file: str,
    *,
    folder: str = "",
    tags: Optional[List[str] | str] = None,
    path: str | None = None,
) -> dict:
    meta = load_source_meta(path)
    entry = {
        "folder": normalize_folder(folder),
        "tags": normalize_tags(tags),
    }
    meta[source_file] = entry
    save_source_meta(meta, path)
    return entry


def get_source_meta(source_file: str, path: str | None = None) -> dict:
    meta = load_source_meta(path)
    entry = meta.get(source_file) or {}
    return {
        "folder": normalize_folder(entry.get("folder", "")),
        "tags": normalize_tags(entry.get("tags", [])),
    }


def delete_source_meta(source_file: str, path: str | None = None) -> bool:
    meta = load_source_meta(path)
    if source_file not in meta:
        return False
    del meta[source_file]
    save_source_meta(meta, path)
    return True


def relative_source_name(path: str, data_dir: str) -> str:
    """data_dir'e göre göreli kaynak adı (klasör/dosya.pdf)."""
    abs_path = os.path.abspath(path)
    abs_data = os.path.abspath(data_dir)
    try:
        rel = os.path.relpath(abs_path, abs_data)
    except ValueError:
        rel = os.path.basename(path)
    return rel.replace("\\", "/")
