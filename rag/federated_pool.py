"""Federated / çok tenant embedding eğitim havuzu."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import METADATA_DIR

from rag.domain_collect import (
    dedupe_pairs,
    load_domain_pairs,
    merge_pairs,
)

DEFAULT_FEDERATED_POOL_PATH = os.path.join(METADATA_DIR, "federated", "training_pool.jsonl")


def _tenant_pair_paths(metadata_root: Optional[str] = None) -> List[Tuple[str, str]]:
    """(tenant_id, pairs.jsonl yolu) listesi."""
    root = metadata_root or METADATA_DIR
    found: List[Tuple[str, str]] = []

    global_path = os.path.join(root, "domain_training", "pairs.jsonl")
    if os.path.isfile(global_path):
        found.append(("global", global_path))

    tenants_root = os.path.join(root, "tenants")
    if os.path.isdir(tenants_root):
        for tenant in sorted(os.listdir(tenants_root)):
            tenant_dir = os.path.join(tenants_root, tenant)
            if not os.path.isdir(tenant_dir):
                continue
            path = os.path.join(tenant_dir, "domain_training", "pairs.jsonl")
            if os.path.isfile(path):
                found.append((tenant, path))

    return found


def load_federated_pool(path: str) -> List[Dict[str, Any]]:
    return load_domain_pairs(path)


def tag_pairs_with_tenant(
    pairs: Sequence[Dict[str, Any]],
    tenant_id: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in pairs:
        item = dict(row)
        item["tenant_id"] = tenant_id
        out.append(item)
    return out


def aggregate_federated_pairs(
    metadata_root: Optional[str] = None,
    *,
    extra_paths: Optional[Sequence[Tuple[str, str]]] = None,
    min_per_tenant: int = 0,
) -> List[Dict[str, Any]]:
    """Tüm tenant domain çiftlerini bir havuzda birleştirir."""
    all_pairs: List[Dict[str, Any]] = []
    paths = list(_tenant_pair_paths(metadata_root))
    if extra_paths:
        paths.extend(list(extra_paths))

    seen_paths: set[str] = set()
    for tenant_id, path in paths:
        abspath = os.path.abspath(path)
        if abspath in seen_paths:
            continue
        seen_paths.add(abspath)
        rows = tag_pairs_with_tenant(load_domain_pairs(path), tenant_id)
        if min_per_tenant > 0 and len(rows) < min_per_tenant:
            continue
        all_pairs.extend(rows)

    return dedupe_pairs(all_pairs)


def save_federated_pool(
    pairs: Sequence[Dict[str, Any]],
    output_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged = dedupe_pairs(pairs)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def summarize_federated_pool(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_tenant: Dict[str, int] = {}
    for row in pairs:
        tid = str(row.get("tenant_id") or "unknown")
        by_tenant[tid] = by_tenant.get(tid, 0) + 1
    return {
        "total": len(pairs),
        "tenants": by_tenant,
        "tenant_count": len(by_tenant),
    }


def build_federated_pool(
    output_path: str,
    *,
    metadata_root: Optional[str] = None,
    min_per_tenant: int = 0,
) -> Dict[str, Any]:
    pairs = aggregate_federated_pairs(
        metadata_root,
        min_per_tenant=min_per_tenant,
    )
    save_federated_pool(pairs, output_path)
    summary = summarize_federated_pool(pairs)
    summary["output"] = output_path
    return summary


def merge_pool_into_training(
    pool_path: str,
    base_pairs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not os.path.isfile(pool_path):
        return list(base_pairs)
    pool = load_federated_pool(pool_path)
    return merge_pairs(base_pairs, pool)
