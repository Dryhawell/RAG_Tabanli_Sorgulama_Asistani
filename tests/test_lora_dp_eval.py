"""LoRA+DP eval yardımcı testleri."""

from rag.lora_dp_eval import lora_adapter_available


def test_lora_adapter_available_missing(tmp_path):
    assert lora_adapter_available(str(tmp_path / "missing")) is False
