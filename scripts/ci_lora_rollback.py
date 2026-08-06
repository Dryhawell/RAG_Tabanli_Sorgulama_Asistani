#!/usr/bin/env python3
"""CI: LoRA rollback önerisi + dry-run veya onaylı rebuild."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    confirm = os.environ.get("CONFIRM_REBUILD", "false").lower() in {"1", "true", "yes"}
    apply_env = os.environ.get("APPLY_ENV_PATCH", "true").lower() in {"1", "true", "yes"}
    lora_dir = os.environ.get("LORA_DIR", "models/lora-dp-embed")
    embedding = os.environ.get("EMBEDDING", "mini-en")

    from rag.lora_dp_eval import (
        execute_lora_rollback,
        lora_adapter_available,
        run_lora_dp_eval_report,
        suggest_lora_rollback,
    )

    os.makedirs("metadata", exist_ok=True)
    payload: dict = {}

    if lora_adapter_available(lora_dir):
        report = run_lora_dp_eval_report(
            embedding,
            lora_dir,
            "evals/fixtures",
            "evals/cases.json",
            report_path="metadata/lora_eval_report.json",
            rollback_path="metadata/lora_rollback.json",
            auto_rollback=True,
            apply_env_patch=apply_env,
            rebuild_dry_run=not confirm,
            rebuild_confirm=confirm,
            env_path="metadata/lora_rollback.env",
            data_dir="data",
            min_accuracy=0.0,
            min_delta=-1.0,
        )
        payload = {
            "mode": "eval",
            "suggestion": report.get("rollback_suggestion"),
            "execution": report.get("rollback_execution"),
            "gate_ok": report.get("gate_ok"),
        }
    else:
        report = {
            "gate_ok": False,
            "lora_ok": False,
            "delta_ok": False,
            "delta_accuracy": -0.1,
            "regressions": [{"case_id": "ci-placeholder"}],
            "lora_dir": lora_dir,
        }
        suggestion = suggest_lora_rollback(report, lora_output_dir=lora_dir)
        # Force rollback suggestion for CI planning when no adapter
        suggestion["should_rollback"] = True
        suggestion["rebuild_embedding"] = embedding
        suggestion["rebuild_command"] = f"python -m rag.cli rebuild --embedding {embedding}"
        suggestion["env_patch"] = {
            "RAG_ENABLE_DOMAIN_EMBEDDING": "0",
            "RAG_DOMAIN_EMBEDDING_MODEL": "",
        }
        report["rollback_suggestion"] = suggestion
        execution = execute_lora_rollback(
            report,
            lora_output_dir=lora_dir,
            env_path="metadata/lora_rollback.env",
            apply_os_environ=apply_env,
            rebuild_dry_run=not confirm,
            rebuild_confirm=confirm,
            data_dir="data",
        )
        with open("metadata/lora_rollback.json", "w", encoding="utf-8") as f:
            json.dump(
                {"suggestion": suggestion, "execution": execution},
                f,
                ensure_ascii=False,
                indent=2,
            )
        payload = {
            "mode": "placeholder",
            "suggestion": suggestion,
            "execution": execution,
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    rebuild = (payload.get("execution") or {}).get("rebuild") or {}
    if confirm and not rebuild.get("executed"):
        print("Confirmed rebuild failed or was skipped", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
