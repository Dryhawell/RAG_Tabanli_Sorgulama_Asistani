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
