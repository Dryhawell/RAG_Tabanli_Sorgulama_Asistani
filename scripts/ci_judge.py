#!/usr/bin/env python3
"""CI: LLM-as-judge / heuristic gate (varsayılan heuristic)."""

from __future__ import annotations

import json
import os
import sys

# Repo kökünü path'e ekle (CI / yerel `python scripts/ci_judge.py`)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    mode = os.environ.get("RAG_JUDGE_MODE", "heuristic").strip().lower() or "heuristic"
    cases = os.environ.get("RAG_JUDGE_CASES", "evals/judge_cases.json")
    min_acc = float(os.environ.get("RAG_JUDGE_MIN_ACCURACY", "1.0"))
    output = os.environ.get("RAG_JUDGE_OUTPUT", "metadata/judge_report.json")
    provider = os.environ.get("RAG_JUDGE_PROVIDER", "ollama")
    model = os.environ.get("RAG_JUDGE_MODEL", "phi3:mini")

    from rag.judge import run_judge_file

    if not os.path.isfile(cases):
        print(f"Judge cases missing: {cases}", file=sys.stderr)
        return 2

    report = run_judge_file(
        cases,
        mode=mode,
        provider=provider,
        model_name=model,
        min_accuracy=min_acc,
    )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report.get("summary") or {}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
