"""Klasör / etiket erişim kontrolü (rol + kullanıcı ACL)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

from rag.auth import User
from rag.meta_store import normalize_folder, normalize_tags
from rag.types import RetrievedChunk


def _has_wildcard(values: Optional[Sequence[str]]) -> bool:
    if values is None:
        return True
    return any(str(v).strip() == "*" for v in values)


def folder_set(user: Optional[User]) -> Optional[Set[str]]:
    """None => tüm klasörler; aksi halde izinli klasör kümesi."""
    if user is None or user.role == "admin":
        return None
    if user.allowed_folders is None or _has_wildcard(user.allowed_folders):
        return None
    return {normalize_folder(f) for f in user.allowed_folders}


def tag_set(user: Optional[User]) -> Optional[Set[str]]:
    """None => tüm etiketler; aksi halde izinli etiket kümesi (lowercase)."""
    if user is None or user.role == "admin":
        return None
    if user.allowed_tags is None or _has_wildcard(user.allowed_tags):
        return None
    return {t.lower() for t in normalize_tags(list(user.allowed_tags))}


def can_access_meta(user: Optional[User], *, folder: str = "", tags: Optional[Sequence[str]] = None) -> bool:
    folders = folder_set(user)
    if folders is not None and normalize_folder(folder) not in folders:
        return False

    allowed_tags = tag_set(user)
    if allowed_tags is None:
        return True
    have = {t.lower() for t in (tags or [])}
    if not have:
        # etiketsiz içerik: klasör izni varsa görülebilir
        return True
    return bool(have & allowed_tags)


def filter_folder_options(user: Optional[User], folders: Sequence[str]) -> List[str]:
    allowed = folder_set(user)
    if allowed is None:
        return list(folders)
    return [f for f in folders if normalize_folder(f) in allowed]


def filter_tag_options(user: Optional[User], tags: Sequence[str]) -> List[str]:
    allowed = tag_set(user)
    if allowed is None:
        return list(tags)
    return [t for t in tags if t.lower() in allowed]


def intersect_folder_filter(
    user: Optional[User],
    ui_folders: Optional[Sequence[str]],
) -> Optional[List[str]]:
    """UI klasör filtresi ile ACL kesişimi."""
    allowed = folder_set(user)
    if allowed is None:
        return list(ui_folders) if ui_folders else None
    if not ui_folders:
        return sorted(allowed)
    out = [normalize_folder(f) for f in ui_folders if normalize_folder(f) in allowed]
    return out


def intersect_tag_filter(
    user: Optional[User],
    ui_tags: Optional[Sequence[str]],
) -> Optional[List[str]]:
    allowed = tag_set(user)
    if allowed is None:
        return list(ui_tags) if ui_tags else None
    if not ui_tags:
        return sorted(allowed)
    out = [t for t in normalize_tags(list(ui_tags)) if t.lower() in allowed]
    return out


def can_ingest_to(
    user: Optional[User],
    *,
    folder: str = "",
    tags: Optional[Sequence[str]] = None,
) -> bool:
    if user is None:
        return True
    if not user.can_ingest:
        return False
    return can_access_meta(user, folder=folder, tags=tags)


def filter_chunks(user: Optional[User], chunks: Iterable[RetrievedChunk]) -> List[RetrievedChunk]:
    return [
        c
        for c in chunks
        if can_access_meta(user, folder=c.metadata.folder, tags=c.metadata.tags)
    ]
