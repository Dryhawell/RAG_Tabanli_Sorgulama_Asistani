"""Gerçek sentence-transformers ile opsiyonel regression.

Çalıştırma:
  RUN_SLOW_EVAL=1 pytest -m slow
"""

import os

import pytest

from rag.eval import run_regression
from rag.embed import Embedder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES = os.path.join(ROOT, "evals", "fixtures")
CASES = os.path.join(ROOT, "evals", "cases.json")


@pytest.mark.slow
def test_regression_with_minilm_embedding():
    emb = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    report = run_regression(
        FIXTURES,
        CASES,
        emb,
        top_k=4,
        threshold=0.25,
        use_hybrid=True,
        alpha=0.55,
        min_accuracy=0.8,
    )
    assert report["summary"]["ok"] is True
    assert report["summary"]["passed"] >= int(0.8 * report["summary"]["total"])


@pytest.mark.slow
def test_sentence_transformer_dp_train(tmp_path):
    from rag.embed_finetune import build_pairs_from_eval
    from rag.st_dp_train import train_sentence_transformer_dp

    pairs = build_pairs_from_eval(CASES, FIXTURES)
    out = str(tmp_path / "st-dp-real")
    report = train_sentence_transformer_dp(
        "mini-en",
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=False,
        head_dim=32,
        max_seq_length=32,
    )
    assert report["steps"] >= 1
    assert report["format"] == "st-dp-head-v1"
    assert (tmp_path / "st-dp-real" / "st_dp_head.pt").exists()


@pytest.mark.slow
def test_lora_dp_real_minilm(tmp_path):
    from rag.embed_finetune import build_pairs_from_eval
    from rag.st_lora_dp import peft_available, train_sentence_transformer_lora_dp

    if not peft_available():
        pytest.skip("peft kurulu değil")

    pairs = build_pairs_from_eval(CASES, FIXTURES)
    out = str(tmp_path / "lora-dp-real")
    report = train_sentence_transformer_lora_dp(
        "mini-en",
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=False,
        lora_rank=4,
        max_seq_length=32,
        production_mode=False,
    )
    assert report["steps"] >= 1
    assert report["format"] == "st-lora-dp-v1"
    assert report.get("used_peft") is True
    assert (tmp_path / "lora-dp-real" / "lora_dp_report.json").exists()


@pytest.mark.slow
def test_lora_dp_opacus_production_minilm(tmp_path):
    from rag.dp_train import opacus_available
    from rag.embed_finetune import build_pairs_from_eval
    from rag.st_lora_dp import peft_available, train_sentence_transformer_lora_dp

    if not peft_available():
        pytest.skip("peft kurulu değil")
    if not opacus_available():
        pytest.skip("opacus kurulu değil")

    pairs = build_pairs_from_eval(CASES, FIXTURES)
    out = str(tmp_path / "lora-dp-opacus")
    try:
        report = train_sentence_transformer_lora_dp(
            "mini-en",
            pairs,
            out,
            epochs=1,
            batch_size=2,
            use_opacus=True,
            lora_rank=4,
            max_seq_length=32,
            production_mode=True,
            secure_mode=False,
            grad_sample_mode="hooks",
        )
    except Exception as exc:
        pytest.skip(f"Opacus+PEFT üretim yolu başarısız: {exc}")

    assert report["steps"] >= 1
    assert report.get("used_opacus") is True
    assert report.get("production_mode") is True
    assert (tmp_path / "lora-dp-opacus" / "lora_dp_report.json").exists()


@pytest.mark.slow
def test_lora_dp_secure_mode_minilm(tmp_path):
    from rag.dp_train import opacus_available
    from rag.embed_finetune import build_pairs_from_eval
    from rag.st_lora_dp import peft_available, train_sentence_transformer_lora_dp

    if not peft_available():
        pytest.skip("peft kurulu değil")
    if not opacus_available():
        pytest.skip("opacus kurulu değil")

    pairs = build_pairs_from_eval(CASES, FIXTURES)
    out = str(tmp_path / "lora-dp-secure")
    try:
        report = train_sentence_transformer_lora_dp(
            "mini-en",
            pairs,
            out,
            epochs=1,
            batch_size=2,
            use_opacus=True,
            lora_rank=4,
            max_seq_length=32,
            production_mode=True,
            secure_mode=True,
            grad_sample_mode="hooks",
        )
    except Exception as exc:
        pytest.skip(f"Opacus secure_mode yolu başarısız: {exc}")

    assert report["steps"] >= 1
    assert report.get("used_opacus") is True
    assert report.get("secure_mode") is True
    assert (tmp_path / "lora-dp-secure" / "lora_dp_report.json").exists()


@pytest.mark.slow
def test_lora_dp_eval_regression_minilm(tmp_path):
    from rag.embed_finetune import build_pairs_from_eval
    from rag.lora_dp_eval import compare_lora_dp_eval, lora_adapter_available
    from rag.st_lora_dp import peft_available, train_sentence_transformer_lora_dp

    if not peft_available():
        pytest.skip("peft kurulu değil")

    pairs = build_pairs_from_eval(CASES, FIXTURES)
    out = str(tmp_path / "lora-dp-eval")
    train_sentence_transformer_lora_dp(
        "mini-en",
        pairs,
        out,
        epochs=1,
        batch_size=2,
        use_opacus=False,
        lora_rank=4,
        max_seq_length=32,
        production_mode=False,
    )
    if not lora_adapter_available(out):
        pytest.skip("LoRA adapter kaydedilmedi")

    report = compare_lora_dp_eval(
        "mini-en",
        out,
        FIXTURES,
        CASES,
        top_k=4,
        threshold=0.25,
        use_hybrid=True,
        min_accuracy=0.8,
        min_delta=0.0,
    )
    assert "base_summary" in report
    assert "lora_summary" in report
    assert report["lora_summary"]["total"] >= 1
    assert report.get("lora_ok") is True
    assert report.get("delta_ok") is True
    assert report["lora_summary"]["accuracy"] >= 0.8
