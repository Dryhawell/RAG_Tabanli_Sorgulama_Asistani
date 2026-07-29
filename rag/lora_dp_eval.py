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


def _case_key(row: Dict[str, Any], idx: int) -> str:
    cid = row.get("case_id")
    if cid:
        return str(cid)
    return f"case-{idx}"


def build_per_case_delta_report(
    report_base: Dict[str, Any],
    report_lora: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Base vs LoRA sonuçlarını case bazında karşılaştırır."""
    base_rows = report_base.get("results") or []
    lora_rows = report_lora.get("results") or []
    lora_map: Dict[str, Dict[str, Any]] = {}
    for i, row in enumerate(lora_rows):
        lora_map[_case_key(row, i)] = row
    deltas: List[Dict[str, Any]] = []
    for i, base_row in enumerate(base_rows):
        key = _case_key(base_row, i)
        lora_row = lora_map.get(key) or {}
        base_pass = bool(base_row.get("passed"))
        lora_pass = bool(lora_row.get("passed"))
        base_gate = float(base_row.get("gate_score") or 0.0)
        lora_gate = float(lora_row.get("gate_score") or 0.0)
        gate_delta = lora_gate - base_gate
        regressed = base_pass and not lora_pass
        improved = not base_pass and lora_pass
        deltas.append(
            {
                "case_id": key,
                "question": base_row.get("question") or lora_row.get("question"),
                "base_passed": base_pass,
                "lora_passed": lora_pass,
                "base_gate": base_gate,
                "lora_gate": lora_gate,
                "gate_delta": gate_delta,
                "regressed": regressed,
                "improved": improved,
                "base_top_sources": base_row.get("top_sources") or [],
                "lora_top_sources": lora_row.get("top_sources") or [],
            }
        )
    return deltas


def list_regressed_cases(per_case: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in per_case if row.get("regressed")]


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
    min_delta: float = 0.0,
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
    delta = f_acc - b_acc
    lora_ok = f_acc + 1e-9 >= float(min_accuracy)
    delta_ok = delta + 1e-9 >= float(min_delta)
    per_case = build_per_case_delta_report(report_base, report_lora)
    regressions = list_regressed_cases(per_case)
    return {
        "base_model": emb_base.model_name,
        "lora_dir": lora_output_dir,
        "lora_model": emb_lora.model_name,
        "base_summary": report_base["summary"],
        "lora_summary": report_lora["summary"],
        "delta_accuracy": delta,
        "improved": f_acc >= b_acc,
        "min_lora_accuracy": float(min_accuracy),
        "min_delta": float(min_delta),
        "lora_ok": lora_ok,
        "delta_ok": delta_ok,
        "gate_ok": lora_ok and delta_ok,
        "base_ok": report_base["summary"].get("ok", True),
        "per_case_deltas": per_case,
        "regressions": regressions,
        "regression_count": len(regressions),
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


def check_lora_delta_gate(report: Dict[str, Any], min_delta: float) -> bool:
    """Base'e göre max düşüş eşiğini kontrol eder (delta_accuracy >= min_delta)."""
    threshold = float(min_delta)
    if report.get("delta_ok") is not None:
        return bool(report["delta_ok"])
    delta = float(report.get("delta_accuracy", 0.0))
    return delta + 1e-9 >= threshold


def check_lora_eval_gates(
    report: Dict[str, Any],
    min_accuracy: float,
    min_delta: float,
) -> bool:
    return check_lora_eval_gate(report, min_accuracy) and check_lora_delta_gate(
        report, min_delta
    )


def run_lora_dp_eval_report(
    base_model: str,
    lora_output_dir: str,
    fixture_dir: str,
    cases_path: str,
    report_path: Optional[str] = None,
    per_case_path: Optional[str] = None,
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
    if per_case_path:
        os.makedirs(os.path.dirname(per_case_path) or ".", exist_ok=True)
        per_case_doc = {
            "per_case_deltas": report.get("per_case_deltas") or [],
            "regressions": report.get("regressions") or [],
            "regression_count": report.get("regression_count", 0),
            "delta_accuracy": report.get("delta_accuracy"),
            "base_model": report.get("base_model"),
            "lora_model": report.get("lora_model"),
        }
        with open(per_case_path, "w", encoding="utf-8") as f:
            json.dump(per_case_doc, f, ensure_ascii=False, indent=2)
    return report
