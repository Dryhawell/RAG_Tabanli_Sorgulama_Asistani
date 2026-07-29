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

