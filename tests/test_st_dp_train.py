"""SentenceTransformer DP train mock testleri."""

from rag.st_dp_train import train_st_dp_mock


def test_train_st_dp_mock(tmp_path):
    pairs = [
        {"anchor": "Soru A uzun?", "positive": "Cevap A yeterince uzun bir metindir."},
        {"anchor": "Soru B uzun?", "positive": "Cevap B yeterince uzun bir metindir."},
        {"anchor": "Soru C uzun?", "positive": "Cevap C yeterince uzun bir metindir."},
        {"anchor": "Soru D uzun?", "positive": "Cevap D yeterince uzun bir metindir."},
    ]
    out = str(tmp_path / "st-dp")
    report = train_st_dp_mock(
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=False,
        seed=3,
    )
    assert report["steps"] >= 1
    assert (tmp_path / "st-dp" / "dp_encoder.pt").exists()
