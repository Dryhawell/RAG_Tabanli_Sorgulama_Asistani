"""LoRA+DP eval yardımcı testleri."""

from rag.lora_dp_eval import (
    check_lora_delta_gate,
    check_lora_eval_gate,
    check_lora_eval_gates,
    lora_adapter_available,
)


def test_lora_adapter_available_missing(tmp_path):
    assert lora_adapter_available(str(tmp_path / "missing")) is False


def test_check_lora_eval_gate_pass():
    report = {
        "lora_ok": True,
        "lora_summary": {"accuracy": 0.9},
    }
    assert check_lora_eval_gate(report, 0.8) is True


def test_check_lora_eval_gate_fail():
    report = {
        "lora_ok": False,
        "lora_summary": {"accuracy": 0.5},
    }
    assert check_lora_eval_gate(report, 0.8) is False


def test_check_lora_eval_gate_zero_threshold():
    report = {"lora_ok": False, "lora_summary": {"accuracy": 0.0}}
    assert check_lora_eval_gate(report, 0.0) is True


def test_check_lora_delta_gate_pass():
    report = {"delta_ok": True, "delta_accuracy": 0.05}
    assert check_lora_delta_gate(report, 0.0) is True


def test_check_lora_delta_gate_fail():
    report = {"delta_ok": False, "delta_accuracy": -0.1}
    assert check_lora_delta_gate(report, 0.0) is False


def test_check_lora_eval_gates_combined():
    report = {
        "lora_ok": True,
        "delta_ok": True,
        "lora_summary": {"accuracy": 0.9},
        "delta_accuracy": 0.02,
    }
    assert check_lora_eval_gates(report, 0.8, 0.0) is True
    report_fail = {"lora_ok": True, "delta_ok": False, "delta_accuracy": -0.05}
    assert check_lora_eval_gates(report_fail, 0.8, 0.0) is False


def test_build_per_case_delta_report():
    from rag.lora_dp_eval import build_per_case_delta_report, list_regressed_cases

    base = {
        "results": [
            {
                "case_id": "a",
                "question": "q1",
                "passed": True,
                "gate_score": 0.9,
                "top_sources": ["x.txt"],
            },
            {
                "case_id": "b",
                "question": "q2",
                "passed": True,
                "gate_score": 0.8,
                "top_sources": ["y.txt"],
            },
        ]
    }
    lora = {
        "results": [
            {
                "case_id": "a",
                "question": "q1",
                "passed": True,
                "gate_score": 0.95,
                "top_sources": ["x.txt"],
            },
            {
                "case_id": "b",
                "question": "q2",
                "passed": False,
                "gate_score": 0.2,
                "top_sources": ["z.txt"],
            },
        ]
    }
    deltas = build_per_case_delta_report(base, lora)
    assert len(deltas) == 2
    regressed = list_regressed_cases(deltas)
    assert len(regressed) == 1
    assert regressed[0]["case_id"] == "b"


def test_run_lora_dp_eval_per_case_output(tmp_path):
    from rag.lora_dp_eval import run_lora_dp_eval_report

    base = {
        "results": [
            {"case_id": "a", "question": "q", "passed": True, "gate_score": 0.9, "top_sources": []},
        ],
        "summary": {"accuracy": 1.0, "ok": True},
    }
    lora = {
        "results": [
            {"case_id": "a", "question": "q", "passed": False, "gate_score": 0.2, "top_sources": []},
        ],
        "summary": {"accuracy": 0.0, "ok": False},
    }

    class FakeEmb:
        model_name = "fake"
        dim = 8

        def encode(self, texts):
            import numpy as np

            return np.zeros((len(texts), 8), dtype=np.float32)

    import rag.lora_dp_eval as lev

    orig_compare = lev.compare_lora_dp_eval

    def fake_compare(*args, **kwargs):
        return {
            "base_model": "b",
            "lora_model": "l",
            "base_summary": base["summary"],
            "lora_summary": lora["summary"],
            "delta_accuracy": -1.0,
            "per_case_deltas": [
                {
                    "case_id": "a",
                    "regressed": True,
                    "base_gate": 0.9,
                    "lora_gate": 0.2,
                }
            ],
            "regressions": [{"case_id": "a"}],
            "regression_count": 1,
        }

    lev.compare_lora_dp_eval = fake_compare
    out_path = str(tmp_path / "per_case.json")
    try:
        run_lora_dp_eval_report(
            "hash",
            "models/x",
            "evals/fixtures",
            "evals/cases.json",
            per_case_path=out_path,
        )
    finally:
        lev.compare_lora_dp_eval = orig_compare
    assert (tmp_path / "per_case.json").exists()
    import json

    with open(out_path, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["regression_count"] == 1
    assert len(doc["per_case_deltas"]) == 1


def test_suggest_lora_rollback():
    from rag.lora_dp_eval import suggest_lora_rollback

    report = {
        "gate_ok": False,
        "lora_ok": False,
        "delta_ok": True,
        "delta_accuracy": -0.2,
        "regressions": [{"case_id": "x"}],
        "lora_dir": "models/lora-dp-embed",
    }
    suggestion = suggest_lora_rollback(report)
    assert suggestion["should_rollback"] is True
    assert suggestion["env_patch"]["RAG_ENABLE_DOMAIN_EMBEDDING"] == "0"
    assert suggestion["rebuild_command"]
    assert any("rebuild" in a for a in suggestion["actions"])

    ok_report = {
        "gate_ok": True,
        "lora_ok": True,
        "delta_ok": True,
        "regressions": [],
    }
    assert suggest_lora_rollback(ok_report)["should_rollback"] is False


def test_apply_lora_rollback_env_and_dry_run(tmp_path):
    from rag.lora_dp_eval import execute_lora_rollback

    report = {
        "gate_ok": False,
        "lora_ok": False,
        "delta_ok": False,
        "regressions": [{"case_id": "z"}],
        "lora_dir": "models/lora-dp-embed",
    }
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("hello", encoding="utf-8")
    env_path = str(tmp_path / "lora_rollback.env")
    result = execute_lora_rollback(
        report,
        env_path=env_path,
        apply_os_environ=True,
        rebuild_dry_run=True,
        data_dir=str(data),
    )
    assert result["env"]["applied"] is True
    assert (tmp_path / "lora_rollback.env").exists()
    assert result["rebuild"]["would_rebuild"] is True
    assert result["rebuild"]["source_count"] >= 1
    assert "rebuild" in result["rebuild"]["command"]


def test_execute_lora_rebuild_requires_confirm():
    from rag.lora_dp_eval import execute_lora_rebuild

    suggestion = {
        "should_rollback": True,
        "rebuild_embedding": "mini-en",
    }
    dry = execute_lora_rebuild(suggestion, confirm=False, data_dir="data")
    assert dry.get("executed") is False
    assert dry.get("reason") == "confirmation_required"

