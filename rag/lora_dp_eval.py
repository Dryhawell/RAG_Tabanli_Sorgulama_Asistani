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


def suggest_lora_rollback(
    report: Dict[str, Any],
    *,
    lora_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Regression / gate başarısızlığında base modele dönüş önerisi üretir."""
    regressions = report.get("regressions") or []
    gate_ok = report.get("gate_ok")
    if gate_ok is None:
        gate_ok = bool(report.get("lora_ok", True)) and bool(report.get("delta_ok", True))
    should_rollback = (not gate_ok) or bool(regressions)
    lora_dir = lora_output_dir or report.get("lora_dir") or ""
    actions = []
    if should_rollback:
        actions.extend(
            [
                "RAG_ENABLE_DOMAIN_EMBEDDING=0 (domain/LoRA preset kapat)",
                "RAG_DOMAIN_EMBEDDING_MODEL= (boşalt; base MiniLM kullan)",
                f"LoRA çıktısını arşivle/ayır: {lora_dir or 'models/lora-dp-embed'}",
                "İndeksi base embedding ile yeniden kur: python -m rag.cli rebuild --embedding mini-en",
            ]
        )
        if regressions:
            actions.append(
                "Regresse eden case'leri incele: "
                + ", ".join(str(r.get("case_id") or "?") for r in regressions[:10])
            )
    return {
        "should_rollback": should_rollback,
        "reason": (
            "gate_failed"
            if not gate_ok
            else ("regressions" if regressions else "ok")
        ),
        "regression_count": len(regressions),
        "delta_accuracy": report.get("delta_accuracy"),
        "lora_ok": report.get("lora_ok"),
        "delta_ok": report.get("delta_ok"),
        "actions": actions,
        "env_patch": {
            "RAG_ENABLE_DOMAIN_EMBEDDING": "0",
            "RAG_DOMAIN_EMBEDDING_MODEL": "",
        }
        if should_rollback
        else {},
        "rebuild_command": "python -m rag.cli rebuild --embedding mini-en"
        if should_rollback
        else None,
        "rebuild_embedding": "mini-en" if should_rollback else None,
    }


def apply_lora_rollback_env_patch(
    suggestion: Dict[str, Any],
    *,
    env_path: Optional[str] = None,
    apply_os_environ: bool = False,
) -> Dict[str, Any]:
    """env_patch'i .env.rollback dosyasına yazar; isteğe bağlı os.environ günceller."""
    patch = dict(suggestion.get("env_patch") or {})
    if not suggestion.get("should_rollback") or not patch:
        return {
            "applied": False,
            "reason": "no_rollback",
            "env_path": env_path,
            "patch": {},
        }
    path = env_path or os.path.join("metadata", "lora_rollback.env")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [f"{k}={v}" for k, v in patch.items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if apply_os_environ:
        for k, v in patch.items():
            os.environ[k] = str(v)
    return {
        "applied": True,
        "env_path": path,
        "patch": patch,
        "os_environ": apply_os_environ,
    }


def dry_run_lora_rebuild(
    suggestion: Dict[str, Any],
    *,
    data_dir: Optional[str] = None,
    embedding: Optional[str] = None,
) -> Dict[str, Any]:
    """Rollback sonrası rebuild'i simüle eder (dosya silmez / indeks yazmaz)."""
    if not suggestion.get("should_rollback"):
        return {"would_rebuild": False, "reason": "no_rollback"}
    emb = embedding or suggestion.get("rebuild_embedding") or "mini-en"
    root = data_dir or "data"
    files: List[str] = []
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith((".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg")):
                    files.append(os.path.join(dirpath, name))
    return {
        "would_rebuild": True,
        "embedding": emb,
        "data_dir": root,
        "source_files": files[:200],
        "source_count": len(files),
        "command": f"python -m rag.cli rebuild --embedding {emb}",
        "note": "Dry-run: indeks oluşturulmadı",
    }


def execute_lora_rebuild(
    suggestion: Dict[str, Any],
    *,
    confirm: bool = False,
    data_dir: Optional[str] = None,
    embedding: Optional[str] = None,
) -> Dict[str, Any]:
    """Onaylı gerçek rebuild. confirm=False ise dry-run döner."""
    dry = dry_run_lora_rebuild(
        suggestion,
        data_dir=data_dir,
        embedding=embedding,
    )
    if not dry.get("would_rebuild"):
        return {**dry, "executed": False}
    if not confirm:
        return {
            **dry,
            "executed": False,
            "reason": "confirmation_required",
            "note": "Onay gerekli: confirm=True veya CI confirm_rebuild",
        }
    emb = dry.get("embedding") or "mini-en"
    root = dry.get("data_dir") or data_dir or "data"
    try:
        from rag.embed import Embedder
        from rag.ingest import rebuild_from_data_dir

        emb_obj = Embedder(model_name=emb)
        index, reports = rebuild_from_data_dir(root, emb_obj)
        return {
            **dry,
            "executed": True,
            "rebuild_report": {
                "files": len(reports or []),
                "index_size": getattr(index, "size", None),
            },
            "note": "Rebuild tamamlandı",
        }
    except Exception as exc:
        return {
            **dry,
            "executed": False,
            "error": str(exc),
            "note": "Rebuild başarısız",
        }


def execute_lora_rollback(
    report: Dict[str, Any],
    *,
    lora_output_dir: Optional[str] = None,
    env_path: Optional[str] = None,
    apply_os_environ: bool = False,
    rebuild_dry_run: bool = True,
    rebuild_confirm: bool = False,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Öneriyi uygular: env patch + dry-run veya onaylı gerçek rebuild."""
    suggestion = suggest_lora_rollback(report, lora_output_dir=lora_output_dir)
    env_result = apply_lora_rollback_env_patch(
        suggestion,
        env_path=env_path,
        apply_os_environ=apply_os_environ,
    )
    rebuild_result = None
    if suggestion.get("should_rollback"):
        if rebuild_confirm:
            rebuild_result = execute_lora_rebuild(
                suggestion,
                confirm=True,
                data_dir=data_dir,
            )
        elif rebuild_dry_run:
            rebuild_result = dry_run_lora_rebuild(
                suggestion,
                data_dir=data_dir,
            )
    return {
        "suggestion": suggestion,
        "env": env_result,
        "rebuild": rebuild_result,
    }


def write_rollback_suggestion(
    report: Dict[str, Any],
    path: str,
    *,
    lora_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    suggestion = suggest_lora_rollback(report, lora_output_dir=lora_output_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suggestion, f, ensure_ascii=False, indent=2)
    return suggestion


def run_lora_dp_eval_report(
    base_model: str,
    lora_output_dir: str,
    fixture_dir: str,
    cases_path: str,
    report_path: Optional[str] = None,
    per_case_path: Optional[str] = None,
    rollback_path: Optional[str] = None,
    auto_rollback: bool = False,
    apply_env_patch: bool = False,
    rebuild_dry_run: bool = False,
    rebuild_confirm: bool = False,
    env_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    report = compare_lora_dp_eval(
        base_model,
        lora_output_dir,
        fixture_dir,
        cases_path,
        **kwargs,
    )
    suggestion = suggest_lora_rollback(report, lora_output_dir=lora_output_dir)
    report["rollback_suggestion"] = suggestion
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
            "rollback_suggestion": suggestion,
        }
        with open(per_case_path, "w", encoding="utf-8") as f:
            json.dump(per_case_doc, f, ensure_ascii=False, indent=2)
    if rollback_path or (auto_rollback and suggestion.get("should_rollback")):
        out = rollback_path or os.path.join(
            lora_output_dir or ".",
            "rollback_suggestion.json",
        )
        write_rollback_suggestion(
            report,
            out,
            lora_output_dir=lora_output_dir,
        )
        report["rollback_path"] = out
    if (
        apply_env_patch or rebuild_dry_run or rebuild_confirm
    ) and suggestion.get("should_rollback"):
        exec_result = execute_lora_rollback(
            report,
            lora_output_dir=lora_output_dir,
            env_path=env_path,
            apply_os_environ=apply_env_patch,
            rebuild_dry_run=rebuild_dry_run and not rebuild_confirm,
            rebuild_confirm=rebuild_confirm,
            data_dir=data_dir,
        )
        report["rollback_execution"] = exec_result
    return report


LORA_PR_COMMENT_MARKER = "<!-- lora-rollback-bot -->"


def build_lora_pr_status(
    suggestion: Optional[Dict[str, Any]] = None,
    *,
    report: Optional[Dict[str, Any]] = None,
    context: str = "lora/rollback",
) -> Dict[str, Any]:
    """GitHub commit status / check özeti üretir."""
    sug = suggestion or (report or {}).get("rollback_suggestion") or {}
    if report and not sug:
        sug = suggest_lora_rollback(report)
    should = bool(sug.get("should_rollback"))
    state = "failure" if should else "success"
    reason = sug.get("reason") or ("gate_failed" if should else "ok")
    desc = (
        f"LoRA rollback önerilir ({reason}, regressions={sug.get('regression_count', 0)})"
        if should
        else "LoRA eval OK — rollback gerekmiyor"
    )
    comment_lines = [
        LORA_PR_COMMENT_MARKER,
        "### LoRA rollback status",
        "",
        f"- **state**: `{state}`",
        f"- **should_rollback**: `{should}`",
        f"- **reason**: `{reason}`",
        f"- **regression_count**: `{sug.get('regression_count', 0)}`",
        f"- **delta_accuracy**: `{sug.get('delta_accuracy')}`",
    ]
    if sug.get("actions"):
        comment_lines.append("- **actions**:")
        for a in sug["actions"]:
            comment_lines.append(f"  - {a}")
    if sug.get("rebuild_command"):
        comment_lines.append(f"- **rebuild**: `{sug['rebuild_command']}`")
    comment_lines.extend(
        [
            "",
            "_Artifact: `metadata/lora_pr_status.json` · workflow `LoRA Rollback Rebuild`_",
        ]
    )
    return {
        "context": context,
        "state": state,
        "description": desc[:140],
        "should_rollback": should,
        "reason": reason,
        "suggestion": sug,
        "comment_marker": LORA_PR_COMMENT_MARKER,
        "comment_markdown": "\n".join(comment_lines) + "\n",
    }


def write_lora_pr_status(
    status: Dict[str, Any],
    *,
    status_path: str = "metadata/lora_pr_status.json",
    comment_path: str = "metadata/lora_pr_comment.md",
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Status JSON + PR yorum markdown + isteğe bağlı GITHUB_STEP_SUMMARY yazar."""
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(comment_path) or ".", exist_ok=True)
    with open(comment_path, "w", encoding="utf-8") as f:
        f.write(status.get("comment_markdown") or "")
    summary = summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(status.get("comment_markdown") or "")
    return {
        "status_path": status_path,
        "comment_path": comment_path,
        "summary_path": summary,
        "state": status.get("state"),
    }
