"""LoRA+DP fine-tuned embedding retrieval eval."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from rag.embed import Embedder, resolve_embedding_model
from rag.eval import run_regression


class LoraEmbedder:
    """SentenceTransformer + PEFT LoRA adapter embedder."""

    def __init__(self, base_model: str, lora_output_dir: str):
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer

        self.base_model = resolve_embedding_model(base_model)
        adapter_dir = os.path.join(lora_output_dir, "lora_adapter")
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"LoRA adapter yok: {adapter_dir}")

        st = SentenceTransformer(self.base_model)
        auto = None
        auto_parent = None
        for mod in st.modules():
            if hasattr(mod, "auto_model"):
                auto = mod.auto_model
                auto_parent = mod
                break
        if auto is None or auto_parent is None:
            raise RuntimeError("SentenceTransformer auto_model bulunamadı")

        peft = PeftModel.from_pretrained(auto, adapter_dir)
        auto_parent.auto_model = peft
        self.model = st
        self.model_name = f"{self.base_model}+lora:{os.path.basename(lora_output_dir)}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


def lora_adapter_available(lora_output_dir: str) -> bool:
    return os.path.isdir(os.path.join(lora_output_dir, "lora_adapter"))


def compare_lora_dp_eval(
    base_model: str,
    lora_output_dir: str,
    fixture_dir: str,
    cases_path: str,
    *,
    top_k: int = 4,
    threshold: float = 0.30,
    use_hybrid: bool = True,
    min_accuracy: float = 0.0,
) -> Dict[str, Any]:
    """Base vs LoRA+DP adapter retrieval karşılaştırması."""
    emb_base = Embedder(model_name=base_model)
    emb_lora = LoraEmbedder(base_model, lora_output_dir)

    report_base = run_regression(
        fixture_dir,
        cases_path,
        emb_base,
        top_k=top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        min_accuracy=min_accuracy,
    )
    report_lora = run_regression(
        fixture_dir,
        cases_path,
        emb_lora,
        top_k=top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        min_accuracy=min_accuracy,
    )

    b_acc = report_base["summary"]["accuracy"]
    f_acc = report_lora["summary"]["accuracy"]
    lora_ok = f_acc + 1e-9 >= float(min_accuracy)
    return {
        "base_model": emb_base.model_name,
        "lora_dir": lora_output_dir,
        "lora_model": emb_lora.model_name,
        "base_summary": report_base["summary"],
        "lora_summary": report_lora["summary"],
        "delta_accuracy": f_acc - b_acc,
        "improved": f_acc >= b_acc,
        "min_lora_accuracy": float(min_accuracy),
        "lora_ok": lora_ok,
        "base_ok": report_base["summary"].get("ok", True),
    }


def check_lora_eval_gate(report: Dict[str, Any], min_accuracy: float) -> bool:
    """LoRA accuracy eşiğini kontrol eder (CI regression gate)."""
    threshold = float(min_accuracy)
    if threshold <= 0:
        return True
    if report.get("lora_ok") is not None:
        return bool(report["lora_ok"])
    lora_acc = float(report.get("lora_summary", {}).get("accuracy", 0.0))
    return lora_acc + 1e-9 >= threshold


def run_lora_dp_eval_report(
    base_model: str,
    lora_output_dir: str,
    fixture_dir: str,
    cases_path: str,
    report_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    report = compare_lora_dp_eval(
        base_model,
        lora_output_dir,
        fixture_dir,
        cases_path,
        **kwargs,
    )
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report
