"""Unfrozen transformer DP-SGD with LoRA adapters (Opacus + PEFT)."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rag.dp_train import estimate_epsilon, opacus_available


def peft_available() -> bool:
    try:
        import peft  # noqa: F401

        return True
    except Exception:
        return False


def _make_tiny_encoder(vocab: int = 128, dim: int = 32, layers: int = 2, heads: int = 4):
    """CI için küçük transformer — LoRA hedefi."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.out_dim = dim

        def forward(self, token_ids):
            x = self.emb(token_ids)
            h = self.encoder(x)
            return F.normalize(h.mean(dim=1), p=2, dim=-1)

    return TinyEnc()


def _attach_manual_lora(module, *, rank: int = 4, alpha: float = 8.0) -> List:
    """PEFT yoksa Linear katmanlara düşük rank adapter ekler."""
    import torch
    import torch.nn as nn

    adapters = []

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.scaling = alpha / rank
            self.A = nn.Parameter(torch.zeros(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self):
            return self.base.in_features

        @property
        def out_features(self):
            return self.base.out_features

        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def replace(parent, name, child):
        # MultiheadAttention içindeki Linear'ları atla (weight getattr / fused path)
        if child.__class__.__name__ == "MultiheadAttention":
            return
        if isinstance(child, nn.Linear) and child.out_features >= rank and child.in_features >= rank:
            lora = LoRALinear(child)
            setattr(parent, name, lora)
            adapters.extend([lora.A, lora.B])
            return
        for n, c in list(child.named_children()):
            replace(child, n, c)

    for n, c in list(module.named_children()):
        replace(module, n, c)
    return adapters


def _hash_tokens(texts: Sequence[str], *, max_len: int = 16, vocab: int = 128):
    import torch
    from rag.hash_embed import HashEmbedder

    # HashEmbedder float üretir; token id için basit hash
    rows = []
    for t in texts:
        ids = []
        for tok in (t or " ").lower().split()[:max_len]:
            ids.append(abs(hash(tok)) % vocab)
        while len(ids) < max_len:
            ids.append(0)
        rows.append(ids[:max_len])
    return torch.tensor(rows, dtype=torch.long)


def train_lora_dp_mock(
    pairs: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 0.05,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
    lora_rank: int = 4,
    seed: int = 7,
) -> Dict[str, Any]:
    """Tiny transformer + manuel LoRA + DP-SGD (model indirmeden)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if not pairs:
        raise ValueError("Eğitim çifti yok")
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    anchors = [str(p.get("anchor") or "") for p in pairs]
    positives = [str(p.get("positive") or "") for p in pairs]
    ta = _hash_tokens(anchors)
    tb = _hash_tokens(positives)
    ds = TensorDataset(ta, tb)
    loader = DataLoader(ds, batch_size=max(1, min(batch_size, len(ds))), shuffle=True)

    model = _make_tiny_encoder()
    for p in model.parameters():
        p.requires_grad = False
    _attach_manual_lora(model, rank=lora_rank)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA parametresi yok")
    optimizer = torch.optim.SGD(trainable, lr=lr)

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
            logits = za @ zb.T
            labels = torch.arange(za.size(0))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if not used_opacus:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        total_norm += float(p.grad.data.norm(2).item() ** 2)
                total_norm = math.sqrt(total_norm) or 1.0
                clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
                for p in model.parameters():
                    if p.grad is not None and p.requires_grad:
                        p.grad.data.mul_(clip_coef)
                        p.grad.data.add_(
                            torch.normal(0.0, noise_multiplier * max_grad_norm, size=p.grad.shape)
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

    weights = os.path.join(output_dir, "lora_dp_mock.pt")
    state = {k.replace("_module.", ""): v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "lora_rank": lora_rank}, weights)
    meta = {
        "format": "lora-dp-mock-v1",
        "pairs": len(pairs),
        "epochs": epochs,
        "steps": steps,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "delta": delta,
        "epsilon": epsilon,
        "used_opacus": used_opacus,
        "used_peft": False,
        "lora_rank": lora_rank,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "weights": weights,
    }
    with open(os.path.join(output_dir, "lora_dp_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def train_sentence_transformer_lora_dp(
    base_model: str,
    pairs: Sequence[Dict[str, Any]],
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    use_opacus: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """SentenceTransformer omurgasına PEFT LoRA + DP-SGD (mümkünse Opacus).

    PEFT yoksa veya LoRA hedefi bulunamazsa mock LoRA yoluna düşer.
    """
    if not pairs:
        raise ValueError("Eğitim çifti yok")
    if not peft_available():
        return train_lora_dp_mock(
            pairs,
            output_dir,
            epochs=epochs,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            delta=delta,
            use_opacus=use_opacus,
            lora_rank=min(4, lora_rank),
            seed=seed,
        )

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader, Dataset

    from rag.embed import resolve_embedding_model

    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    model_name = resolve_embedding_model(base_model)
    st = SentenceTransformer(model_name)
    st.max_seq_length = max_seq_length

    # AutoModel omurgasını bul
    auto = None
    for mod in st.modules():
        if mod.__class__.__name__ in {"Transformer", "WordEmbeddings"}:
            continue
        if hasattr(mod, "auto_model"):
            auto = mod.auto_model
            break
    if auto is None:
        # fallback
        return train_lora_dp_mock(
            pairs,
            output_dir,
            epochs=epochs,
            batch_size=batch_size,
            use_opacus=use_opacus,
            lora_rank=min(4, lora_rank),
            seed=seed,
        )

    for p in auto.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query", "value", "key", "dense"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    try:
        auto = get_peft_model(auto, lora_cfg)
    except Exception:
        # hedef modül adları modele göre değişebilir
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        try:
            auto = get_peft_model(auto, lora_cfg)
        except Exception:
            return train_lora_dp_mock(
                pairs,
                output_dir,
                epochs=epochs,
                batch_size=batch_size,
                use_opacus=use_opacus,
                lora_rank=min(4, lora_rank),
                seed=seed,
            )

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
        # encode with gradients through peft model by using tokenizer + auto
        tok = st.tokenizer
        ba = tok(
            list(anchors),
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        bb = tok(
            list(positives),
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        return ba, bb

    def pooled(batch_inputs):
        out = auto(**{k: v for k, v in batch_inputs.items()})
        hidden = out.last_hidden_state
        mask = batch_inputs["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
        return F.normalize(emb, p=2, dim=-1)

    ds = PairDS(pairs)
    loader = DataLoader(
        ds,
        batch_size=max(1, min(batch_size, len(ds))),
        shuffle=True,
        collate_fn=collate,
    )
    trainable = [p for p in auto.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # Opacus + PEFT çoğu mimaride kırılgan; varsayılan manuel DP
    used_opacus = False
    privacy_engine = None
    if use_opacus and opacus_available():
        try:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()
            auto, optimizer, loader = privacy_engine.make_private(
                module=auto,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            used_opacus = True
        except Exception:
            used_opacus = False
            privacy_engine = None

    steps = 0
    losses: List[float] = []
    for _ in range(max(1, epochs)):
        for ba, bb in loader:
            optimizer.zero_grad()
            za = pooled(ba)
            zb = pooled(bb)
            logits = za @ zb.T
            labels = torch.arange(za.size(0), device=za.device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            if not used_opacus:
                total_norm = 0.0
                for p in auto.parameters():
                    if p.grad is not None and p.requires_grad:
                        total_norm += float(p.grad.data.norm(2).item() ** 2)
                total_norm = math.sqrt(total_norm) or 1.0
                clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
                for p in auto.parameters():
                    if p.grad is not None and p.requires_grad:
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

    adapter_dir = os.path.join(output_dir, "lora_adapter")
    try:
        # GradSampleModule wrapper olabilir
        to_save = auto._module if hasattr(auto, "_module") else auto
        to_save.save_pretrained(adapter_dir)
    except Exception:
        adapter_dir = None
        torch.save(auto.state_dict(), os.path.join(output_dir, "lora_state.pt"))

    meta = {
        "format": "st-lora-dp-v1",
        "base_model": model_name,
        "pairs": len(pairs),
        "epochs": epochs,
        "steps": steps,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "delta": delta,
        "epsilon": epsilon,
        "used_opacus": used_opacus,
        "used_peft": True,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "avg_loss": float(np.mean(losses)) if losses else None,
        "adapter_dir": adapter_dir,
    }
    with open(os.path.join(output_dir, "lora_dp_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def train_lora_dp_from_pairs_file(
    base_model: str,
    pairs_path: str,
    output_dir: str,
    *,
    mock: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    from rag.embed_finetune import load_pairs_jsonl

    pairs = load_pairs_jsonl(pairs_path)
    if mock:
        return train_lora_dp_mock(pairs, output_dir, **kwargs)
    return train_sentence_transformer_lora_dp(base_model, pairs, output_dir, **kwargs)
