"""SentenceTransformer + Opacus DP fine-tune (projeksiyon başı)."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rag.dp_train import estimate_epsilon, opacus_available


def _freeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def train_sentence_transformer_dp(
    base_model: str,
    pairs: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
    head_dim: int = 64,
    max_seq_length: int = 64,
    seed: int = 42,
    freeze_backbone: bool = True,
) -> Dict[str, Any]:
    """SentenceTransformer omurgasını dondurup DP-SGD ile projeksiyon başı eğitir.

    Tam model + Opacus zor olduğu için pratik yaklaşım: frozen encoder + trainable Dense head.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader, Dataset

    if not pairs:
        raise ValueError("Eğitim çifti yok")

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    from rag.embed import resolve_embedding_model

    model_name = resolve_embedding_model(base_model)
    st = SentenceTransformer(model_name)
    st.max_seq_length = max_seq_length
    if freeze_backbone:
        _freeze_module(st)

    in_dim = st.get_sentence_embedding_dimension()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(in_dim, head_dim),
                nn.Tanh(),
                nn.Linear(head_dim, head_dim),
            )

        def forward(self, x):
            return F.normalize(self.proj(x), p=2, dim=-1)

    head = Head()

    class PairDS(Dataset):
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            r = self.rows[idx]
            return str(r.get("anchor") or ""), str(r.get("positive") or "")

    def collate(batch):
        anchors, positives = zip(*batch)
        with torch.no_grad():
            ea = st.encode(
                list(anchors),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            eb = st.encode(
                list(positives),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return ea.float(), eb.float()

    ds = PairDS(pairs)
    loader = DataLoader(
        ds,
        batch_size=max(1, min(batch_size, len(ds))),
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    privacy_engine = None
    used_opacus = False
    if use_opacus and opacus_available():
        try:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()
            head, optimizer, loader = privacy_engine.make_private(
                module=head,
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
            za = head(xa)
            zb = head(xb)
            logits = za @ zb.T
            labels = torch.arange(za.size(0), device=za.device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if not used_opacus:
                total_norm = 0.0
                for p in head.parameters():
                    if p.grad is not None:
                        total_norm += float(p.grad.data.norm(2).item() ** 2)
                total_norm = math.sqrt(total_norm) or 1.0
                clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
                for p in head.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
                        p.grad.data.add_(
                            torch.normal(
                                0.0,
                                noise_multiplier * max_grad_norm,
                                size=p.grad.shape,
                                device=p.grad.device,
                            )
                        )
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().item()))

    sample_rate = min(1.0, max(1, batch_size) / max(1, len(ds)))
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

    state = head.state_dict()
    clean_state = {k.replace("_module.", ""): v.detach().cpu() for k, v in state.items()}
    weights_path = os.path.join(output_dir, "st_dp_head.pt")
    torch.save(
        {
            "state_dict": clean_state,
            "base_model": model_name,
            "in_dim": in_dim,
            "head_dim": head_dim,
            "freeze_backbone": freeze_backbone,
        },
        weights_path,
    )
    # SentenceTransformer kaydı (omurga) — head ayrı
    st_dir = os.path.join(output_dir, "sentence-transformer")
    try:
        st.save(st_dir)
    except Exception:
        st_dir = None

    meta = {
        "format": "st-dp-head-v1",
        "base_model": model_name,
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
        "sentence_transformer_dir": st_dir,
        "freeze_backbone": freeze_backbone,
        "head_dim": head_dim,
    }
    with open(os.path.join(output_dir, "st_dp_train_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def train_sentence_transformer_dp_from_pairs_file(
    base_model: str,
    pairs_path: str,
    output_dir: str,
    **kwargs,
) -> Dict[str, Any]:
    from rag.embed_finetune import load_pairs_jsonl

    pairs = load_pairs_jsonl(pairs_path)
    return train_sentence_transformer_dp(base_model, pairs, output_dir, **kwargs)


def train_st_dp_mock(
    pairs: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
    dim: int = 32,
    seed: int = 0,
) -> Dict[str, Any]:
    """Model indirmeden ST+DP akışını taklit eder (hash özellik + head)."""
    from rag.dp_train import train_dp_sgd_hash_encoder

    return train_dp_sgd_hash_encoder(
        pairs,
        output_dir,
        dim=dim,
        epochs=epochs,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        delta=delta,
        use_opacus=use_opacus,
        seed=seed,
    )
