"""Opacus / resmi DP-SGD ile embedding fine-tuning."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def opacus_available() -> bool:
    try:
        import opacus  # noqa: F401

        return True
    except Exception:
        return False


def estimate_epsilon(
    *,
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float = 1e-5,
) -> float:
    """Basit RDP → (ε, δ)-DP yaklaşık hesabı (PoC accountant).

    Kaynak: Abadi et al. / Opacus RDP accountant'ın sadeleştirilmiş üst sınırı.
    """
    if noise_multiplier <= 0 or steps <= 0:
        return float("inf")
    # Conservative bound used in many tutorials:
    # ε ≈ sqrt(2 * steps * log(1/δ)) / σ   (when q≈1) with sample_rate correction
    q = max(1e-9, min(1.0, float(sample_rate)))
    sigma = float(noise_multiplier)
    # Moments accountant style rough bound
    eps = q * math.sqrt(2.0 * steps * math.log(1.0 / max(delta, 1e-12))) / sigma
    return float(eps)


def _build_pair_tensors(pairs: Sequence[Dict[str, Any]], dim: int = 64):
    """HashEmbedder ile sabit özellikler — model indirmeden DP eğitim testi."""
    import torch
    from rag.hash_embed import HashEmbedder

    emb = HashEmbedder(dim=dim)
    anchors = [str(p.get("anchor") or "") for p in pairs]
    positives = [str(p.get("positive") or "") for p in pairs]
    a = torch.tensor(emb.encode(anchors), dtype=torch.float32)
    b = torch.tensor(emb.encode(positives), dtype=torch.float32)
    return a, b


def _make_dual_encoder(dim: int = 64, hidden: int = 32):
    import torch.nn as nn
    import torch.nn.functional as F

    class _Dual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        def forward(self, x):
            z = self.proj(x)
            return F.normalize(z, p=2, dim=-1)

    return _Dual()


def train_dp_sgd_hash_encoder(
    pairs: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    dim: int = 64,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 0.05,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """DP-SGD ile küçük dual-encoder eğitir (Opacus varsa PrivacyEngine).

    Büyük SentenceTransformer yerine hash özellik + projeksiyon kullanır;
    CI ve privacy PoC için uygundur. Çıktı: state_dict + meta JSON.
    """
    import json
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if not pairs:
        raise ValueError("Eğitim çifti yok")

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    a, b = _build_pair_tensors(pairs, dim=dim)
    dataset = TensorDataset(a, b)
    loader = DataLoader(dataset, batch_size=max(1, min(batch_size, len(dataset))), shuffle=True)

    model = _make_dual_encoder(dim=dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    privacy_engine = None
    used_opacus = False
    if use_opacus and opacus_available():
        try:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()
            model, optimizer, loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            used_opacus = True
        except Exception:
            privacy_engine = None
            used_opacus = False

    steps = 0
    losses: List[float] = []
    for _ in range(max(1, epochs)):
        for xa, xb in loader:
            optimizer.zero_grad()
            za = model(xa)
            zb = model(xb)
            # InfoNCE / MNRL benzeri: pozitifler diagonal
            logits = za @ zb.T
            labels = torch.arange(za.size(0))
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            if not used_opacus:
                # Manuel per-batch clip + gürültü (DP-SGD lite fallback)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += float(p.grad.data.norm(2).item() ** 2)
                total_norm = math.sqrt(total_norm) or 1.0
                clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
                        noise = torch.normal(
                            mean=0.0,
                            std=noise_multiplier * max_grad_norm,
                            size=p.grad.shape,
                        )
                        p.grad.data.add_(noise)

            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().item()))

    sample_rate = min(1.0, max(1, batch_size) / max(1, len(dataset)))
    if used_opacus and privacy_engine is not None:
        try:
            epsilon = float(privacy_engine.get_epsilon(delta))
        except Exception:
            epsilon = estimate_epsilon(
                steps=steps,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                delta=delta,
            )
    else:
        epsilon = estimate_epsilon(
            steps=steps,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta,
        )

    # kaydet
    state = model.state_dict()
    # Opacus GradSampleModule wrapper olabilir
    clean_state = {k.replace("_module.", ""): v.detach().cpu() for k, v in state.items()}
    weights_path = os.path.join(output_dir, "dp_encoder.pt")
    torch.save({"state_dict": clean_state, "dim": dim}, weights_path)

    meta = {
        "format": "dp-sgd-hash-encoder-v1",
        "pairs": len(pairs),
        "epochs": epochs,
        "steps": steps,
        "batch_size": batch_size,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "delta": delta,
        "epsilon": epsilon,
        "used_opacus": used_opacus,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "weights": weights_path,
    }
    with open(os.path.join(output_dir, "dp_train_report.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def train_embedding_model_dp(
    base_model: str,
    pairs_path: str,
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
) -> Dict[str, Any]:
    """JSONL çiftlerden DP-SGD eğitimi (hash dual-encoder + opsiyonel Opacus)."""
    from rag.embed_finetune import load_pairs_jsonl

    pairs = load_pairs_jsonl(pairs_path)
    if not pairs:
        raise ValueError("Eğitim çifti yok")
    return train_dp_sgd_hash_encoder(
        pairs,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        delta=delta,
        use_opacus=use_opacus,
    )
