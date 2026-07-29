"""LoRA+DP eval yardımcı testleri."""

from rag.lora_dp_eval import check_lora_eval_gate, lora_adapter_available


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

