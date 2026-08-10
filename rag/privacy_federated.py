"""Cross-tenant embedding privacy: DP gürültü + basit şifreli aggregation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.config import METADATA_DIR

DEFAULT_PRIVATE_POOL_PATH = os.path.join(
    METADATA_DIR, "federated", "private_training_pool.jsonl"
)


def _l2_clip(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n > max_norm and n > 0:
        return (vec * (max_norm / n)).astype(np.float32)
    return vec.astype(np.float32)


def add_gaussian_noise(
    vec: np.ndarray,
    *,
    noise_multiplier: float,
    clip_norm: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """DP-SGD benzeri: L2 clip + Gaussian gürültü."""
    clipped = _l2_clip(np.asarray(vec, dtype=np.float32), clip_norm)
    rng = np.random.default_rng(seed)
    sigma = float(noise_multiplier) * float(clip_norm)
    noise = rng.normal(0.0, sigma, size=clipped.shape).astype(np.float32)
    return clipped + noise


def hash_text(text: str, *, salt: str = "") -> str:
    raw = f"{salt}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def redact_pair(
    pair: Dict[str, Any],
    *,
    salt: str = "",
    keep_tenant: bool = True,
) -> Dict[str, Any]:
    """Metinleri hash'e çevirerek cross-tenant sızıntıyı azaltır (eğitim için embedding gerekir)."""
    out: Dict[str, Any] = {
        "anchor_hash": hash_text(str(pair.get("anchor") or ""), salt=salt),
        "positive_hash": hash_text(str(pair.get("positive") or ""), salt=salt),
        "source": pair.get("source") or "redacted",
    }
    if keep_tenant and pair.get("tenant_id"):
        out["tenant_id"] = pair["tenant_id"]
    return out


def pair_contribution_vector(
    pair: Dict[str, Any],
    *,
    dim: int = 64,
) -> np.ndarray:
    """Deterministik hash embedding — tenant katkısını vektör olarak temsil eder."""
    from rag.hash_embed import HashEmbedder

    emb = HashEmbedder(dim=dim)
    anchor = str(pair.get("anchor") or "")
    positive = str(pair.get("positive") or "")
    vecs = emb.encode([anchor, positive])
    # ortalama katkı
    return ((vecs[0] + vecs[1]) * 0.5).astype(np.float32)


def privatize_contribution(
    pair: Dict[str, Any],
    *,
    dim: int = 64,
    noise_multiplier: float = 1.0,
    clip_norm: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    vec = pair_contribution_vector(pair, dim=dim)
    noisy = add_gaussian_noise(
        vec,
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        seed=seed,
    )
    return {
        "tenant_id": pair.get("tenant_id") or "unknown",
        "vector": noisy.tolist(),
        "dim": dim,
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
        "anchor_preview": str(pair.get("anchor") or "")[:48],
    }


def secure_aggregate_vectors(
    contributions: Sequence[Dict[str, Any]],
    *,
    shared_secret: str = "",
) -> Dict[str, Any]:
    """Basit 'şifreli' aggregation: HMAC maskeleri + ortalama (PoC).

    Her katkıya sunucu/tenant sırlarından türetilen maske eklenir;
    aggregation sonrası maskeler iptal edilir (aynı secret ile).
    Gerçek Secure Aggregation yerine deterministik PoC.
    """
    if not contributions:
        return {"count": 0, "vector": [], "dim": 0}

    dim = int(contributions[0].get("dim") or len(contributions[0].get("vector") or []))
    acc = np.zeros(dim, dtype=np.float32)
    masks = []
    for i, row in enumerate(contributions):
        vec = np.asarray(row.get("vector") or [], dtype=np.float32)
        if vec.shape[0] != dim:
            continue
        mask = _hmac_mask(shared_secret or "rag-federated", f"{row.get('tenant_id')}:{i}", dim)
        masks.append(mask)
        acc += vec + mask

    n = len(masks) or 1
    # maskeleri çıkar
    for mask in masks:
        acc -= mask
    avg = acc / float(n)
    return {
        "count": n,
        "dim": dim,
        "vector": avg.tolist(),
        "tenants": sorted({str(r.get("tenant_id") or "?") for r in contributions}),
    }


def _hmac_mask(secret: str, key: str, dim: int) -> np.ndarray:
    digest = hmac.new(secret.encode("utf-8"), key.encode("utf-8"), hashlib.sha256).digest()
    vals: List[float] = []
    counter = 0
    while len(vals) < dim:
        block = hmac.new(
            secret.encode("utf-8"),
            digest + counter.to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        for i in range(0, len(block), 4):
            if len(vals) >= dim:
                break
            # unsigned 32-bit -> [0,1)
            n = int.from_bytes(block[i : i + 4], "big")
            vals.append((n / 2**32) * 2.0 - 1.0)
        counter += 1
    arr = np.asarray(vals[:dim], dtype=np.float32)
    n = float(np.linalg.norm(arr)) or 1.0
    return (arr / n * 0.01).astype(np.float32)


def build_private_federated_pool(
    pairs: Sequence[Dict[str, Any]],
    output_path: str,
    *,
    noise_multiplier: float = 1.0,
    clip_norm: float = 1.0,
    dim: int = 64,
    shared_secret: Optional[str] = None,
    keep_text_pairs: bool = True,
) -> Dict[str, Any]:
    """Tenant çiftlerinden DP katkılar + secure aggregate özeti üretir.

    keep_text_pairs=True ise eğitim için metin çiftleri de yazılır
    (yalnızca local pipeline; production'da False tercih edilebilir).
    """
    secret = shared_secret or os.getenv("RAG_FEDERATED_SECRET", "rag-federated")
    contributions = []
    for i, pair in enumerate(pairs):
        contributions.append(
            privatize_contribution(
                pair,
                dim=dim,
                noise_multiplier=noise_multiplier,
                clip_norm=clip_norm,
                seed=1000 + i,
            )
        )
    aggregate = secure_aggregate_vectors(contributions, shared_secret=secret)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "format": "private-federated-v1",
        "aggregate": aggregate,
        "contributions_count": len(contributions),
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
        "privacy": "dp-sgd-lite+secure-agg-poc",
    }
    if keep_text_pairs:
        # eğitim için metin korunur; DP aggregate ayrı raporlanır
        payload["pairs"] = [dict(p) for p in pairs]

    report_path = output_path
    if output_path.endswith(".jsonl"):
        # JSONL: her satır bir pair veya aggregate meta
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", **{k: v for k, v in payload.items() if k != "pairs"}}, ensure_ascii=False) + "\n")
            if keep_text_pairs:
                for p in pairs:
                    f.write(json.dumps({"type": "pair", **p}, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "output": report_path,
        "pairs": len(pairs),
        "contributions": len(contributions),
        "aggregate_count": aggregate.get("count"),
        "tenants": aggregate.get("tenants") or [],
        "noise_multiplier": noise_multiplier,
        "clip_norm": clip_norm,
    }


def load_private_pool_pairs(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("pairs") or [])
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "pair" and row.get("anchor") and row.get("positive"):
                rows.append(row)
            elif row.get("anchor") and row.get("positive") and row.get("type") is None:
                rows.append(row)
    return rows
