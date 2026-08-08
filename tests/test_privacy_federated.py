"""Cross-tenant privacy (DP-SGD lite) testleri."""

import numpy as np

from rag.privacy_federated import (
    add_gaussian_noise,
    build_private_federated_pool,
    load_private_pool_pairs,
    privatize_contribution,
    redact_pair,
    secure_aggregate_vectors,
)


def test_add_gaussian_noise_deterministic_seed():
    v = np.ones(8, dtype=np.float32)
    a = add_gaussian_noise(v, noise_multiplier=0.5, clip_norm=1.0, seed=7)
    b = add_gaussian_noise(v, noise_multiplier=0.5, clip_norm=1.0, seed=7)
    assert np.allclose(a, b)
    assert a.shape == (8,)


def test_privatize_and_secure_aggregate():
    pairs = [
        {
            "anchor": "İzin kaç gün?",
            "positive": "Çalışanlar 14 gün ücretli izin hakkına sahiptir.",
            "tenant_id": "acme",
        },
        {
            "anchor": "Stok kodu nedir?",
            "positive": "Atlas Not Defteri stok kodu AT-ND-001 olarak kayıtlıdır.",
            "tenant_id": "beta",
        },
    ]
    contribs = [privatize_contribution(p, dim=32, noise_multiplier=0.2, seed=i) for i, p in enumerate(pairs)]
    agg = secure_aggregate_vectors(contribs, shared_secret="test-secret")
    assert agg["count"] == 2
    assert len(agg["vector"]) == 32
    assert set(agg["tenants"]) == {"acme", "beta"}


def test_build_private_pool_jsonl(tmp_path):
    pairs = [
        {
            "anchor": "Soru bir?",
            "positive": "Cevap metni yeterince uzun olmalıdır.",
            "tenant_id": "t1",
        }
    ]
    out = str(tmp_path / "private.jsonl")
    report = build_private_federated_pool(
        pairs,
        out,
        noise_multiplier=0.5,
        clip_norm=1.0,
        keep_text_pairs=True,
    )
    assert report["pairs"] == 1
    loaded = load_private_pool_pairs(out)
    assert len(loaded) == 1
    assert loaded[0]["anchor"] == "Soru bir?"


def test_redact_pair():
    red = redact_pair({"anchor": "gizli", "positive": "metin", "tenant_id": "x"}, salt="s")
    assert "anchor_hash" in red
    assert "gizli" not in str(red)
    assert red["tenant_id"] == "x"
