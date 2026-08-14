"""LoRA + DP-SGD mock testleri."""

from rag.st_lora_dp import peft_available, train_lora_dp_mock


def test_train_lora_dp_mock(tmp_path):
    pairs = [
        {"anchor": "Soru A uzun?", "positive": "Cevap A yeterince uzun bir metindir."},
        {"anchor": "Soru B uzun?", "positive": "Cevap B yeterince uzun bir metindir."},
        {"anchor": "Soru C uzun?", "positive": "Cevap C yeterince uzun bir metindir."},
        {"anchor": "Soru D uzun?", "positive": "Cevap D yeterince uzun bir metindir."},
    ]
    out = str(tmp_path / "lora-dp")
    report = train_lora_dp_mock(
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=False,
        lora_rank=4,
        seed=5,
    )
    assert report["steps"] >= 1
    assert report["format"] == "lora-dp-mock-v1"
    assert report["used_peft"] is False
    assert (tmp_path / "lora-dp" / "lora_dp_mock.pt").exists()
    assert (tmp_path / "lora-dp" / "lora_dp_report.json").exists()


def test_peft_available_bool():
    assert isinstance(peft_available(), bool)
