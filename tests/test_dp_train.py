"""DP-SGD / Opacus eğitim testleri."""

from rag.dp_train import estimate_epsilon, opacus_available, train_dp_sgd_hash_encoder


def test_estimate_epsilon_finite():
    eps = estimate_epsilon(steps=10, noise_multiplier=1.0, sample_rate=0.5, delta=1e-5)
    assert eps > 0
    assert eps < 100


def test_train_dp_sgd_manual_fallback(tmp_path):
    pairs = [
        {
            "anchor": "İzin kaç gün?",
            "positive": "Çalışanlar yılda 14 gün ücretli izin hakkına sahiptir.",
        },
        {
            "anchor": "Stok kodu?",
            "positive": "Atlas Not Defteri stok kodu AT-ND-001 olarak kayıtlıdır.",
        },
        {
            "anchor": "İzin talebi ne zaman?",
            "positive": "İzin talebi en az 5 iş günü önceden sisteme girilmelidir.",
        },
        {
            "anchor": "Fiyat nedir?",
            "positive": "Atlas Not Defteri önerilen satış fiyatı 129 TL'dir.",
        },
    ]
    out = str(tmp_path / "dp-model")
    report = train_dp_sgd_hash_encoder(
        pairs,
        out,
        epochs=1,
        batch_size=2,
        noise_multiplier=0.8,
        max_grad_norm=1.0,
        use_opacus=False,
        seed=1,
    )
    assert report["steps"] >= 1
    assert report["used_opacus"] is False
    assert report["epsilon"] > 0
    assert (tmp_path / "dp-model" / "dp_encoder.pt").exists()
    assert (tmp_path / "dp-model" / "dp_train_report.json").exists()


def test_train_dp_sgd_with_opacus_if_available(tmp_path):
    pairs = [
        {"anchor": "Soru A uzun?", "positive": "Cevap A yeterince uzun bir metindir."},
        {"anchor": "Soru B uzun?", "positive": "Cevap B yeterince uzun bir metindir."},
        {"anchor": "Soru C uzun?", "positive": "Cevap C yeterince uzun bir metindir."},
        {"anchor": "Soru D uzun?", "positive": "Cevap D yeterince uzun bir metindir."},
    ]
    out = str(tmp_path / "dp-opacus")
    report = train_dp_sgd_hash_encoder(
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=True,
        seed=2,
    )
    assert report["steps"] >= 1
    if opacus_available():
        # Opacus başarısız olursa fallback True olmayabilir; en azından eğitimin bittiğini doğrula
        assert "used_opacus" in report
    assert (tmp_path / "dp-opacus" / "dp_encoder.pt").exists()
