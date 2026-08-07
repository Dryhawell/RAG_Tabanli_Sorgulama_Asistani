import json
import os

from rag.cli import build_parser, main
from rag.eval import load_cases, run_regression
from rag.hash_embed import HashEmbedder


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES = os.path.join(ROOT, "evals", "fixtures")
CASES = os.path.join(ROOT, "evals", "cases.json")


def test_load_cases_from_repo():
    cases = load_cases(CASES)
    assert len(cases) >= 4
    assert any(c.expect_no_answer for c in cases)
    assert any(c.expected_source for c in cases)


def test_regression_passes_with_hash_embedder():
    report = run_regression(
        FIXTURES,
        CASES,
        HashEmbedder(dim=64),
        top_k=4,
        threshold=0.30,
        use_hybrid=True,
        alpha=0.5,
        min_accuracy=1.0,
    )
    assert report["summary"]["ok"] is True
    assert report["summary"]["failed"] == 0


def test_cli_eval_command(tmp_path):
    out = tmp_path / "report.json"
    code = main(
        [
            "eval",
            "--fixtures",
            FIXTURES,
            "--cases",
            CASES,
            "--embedding",
            "hash",
                "--threshold",
                "0.30",
                "--min-accuracy",
                "1.0",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["passed"] == data["summary"]["total"]


def test_cli_parser_has_eval():
    parser = build_parser()
    args = parser.parse_args(["eval", "--embedding", "hash", "--min-accuracy", "0.8"])
    assert args.command == "eval"
    assert args.embedding == "hash"
    assert args.min_accuracy == 0.8
