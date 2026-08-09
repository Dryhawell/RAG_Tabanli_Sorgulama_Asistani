"""Federated embedding havuzu testleri."""

import json

from rag.domain_collect import append_domain_pairs
from rag.federated_pool import (
    aggregate_federated_pairs,
    build_federated_pool,
    summarize_federated_pool,
    tag_pairs_with_tenant,
)


def test_aggregate_federated_from_tenants(tmp_path):
    meta = tmp_path / "metadata"
    t1 = meta / "tenants" / "acme" / "domain_training"
    t2 = meta / "tenants" / "beta" / "domain_training"
    t1.mkdir(parents=True)
    t2.mkdir(parents=True)
    append_domain_pairs(
        [{"anchor": "Soru A uzun?", "positive": "Cevap metni tenant acme için uzun"}],
        str(t1 / "pairs.jsonl"),
    )
    append_domain_pairs(
        [{"anchor": "Soru B uzun?", "positive": "Cevap metni tenant beta için uzun"}],
        str(t2 / "pairs.jsonl"),
    )
    pairs = aggregate_federated_pairs(str(meta))
    assert len(pairs) == 2
    tenants = {p.get("tenant_id") for p in pairs}
    assert tenants == {"acme", "beta"}


def test_build_federated_pool_summary(tmp_path):
    meta = tmp_path / "metadata"
    global_dir = meta / "domain_training"
    global_dir.mkdir(parents=True)
    append_domain_pairs(
        [{"anchor": "Global soru?", "positive": "Global cevap metni uzun örnek"}],
        str(global_dir / "pairs.jsonl"),
    )
    out = str(meta / "federated" / "pool.jsonl")
    summary = build_federated_pool(out, metadata_root=str(meta))
    assert summary["total"] >= 1
    assert summary["tenant_count"] >= 1
    assert summary["output"] == out


def test_tag_pairs_with_tenant():
    tagged = tag_pairs_with_tenant(
        [{"anchor": "q", "positive": "long positive text"}],
        "tenant-x",
    )
    assert tagged[0]["tenant_id"] == "tenant-x"


def test_summarize_federated_pool():
    rows = tag_pairs_with_tenant(
        [
            {"anchor": "a1", "positive": "p1 long enough"},
            {"anchor": "a2", "positive": "p2 long enough"},
        ],
        "t1",
    )
    summary = summarize_federated_pool(rows)
    assert summary["total"] == 2
    assert summary["tenants"]["t1"] == 2
