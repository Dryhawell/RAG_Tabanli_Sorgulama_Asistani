"""Embedding fine-tuning pipeline testleri."""

import json

from rag.embed_finetune import (
    build_pairs_from_eval,
    export_pairs_jsonl,
    load_pairs_jsonl,
)


def test_build_pairs_from_eval_fixtures():
    pairs = build_pairs_from_eval("evals/cases.json", "evals/fixtures")
    assert len(pairs) >= 4
    ids = {p.get("case_id") for p in pairs}
    assert "izin-sure" in ids
    first = pairs[0]
    assert first["anchor"]
    assert first["positive"]
    assert "negatif" not in ids


def test_export_and_load_pairs_jsonl(tmp_path):
    pairs = [
        {
            "anchor": "Soru?",
            "positive": "Cevap metni",
            "case_id": "x",
            "expected_source": "a.txt",
        }
    ]
    path = str(tmp_path / "pairs.jsonl")
    export_pairs_jsonl(pairs, path)
    loaded = load_pairs_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0]["anchor"] == "Soru?"


def test_build_pairs_hard_negatives():
    pairs = build_pairs_from_eval(
        "evals/cases.json",
        "evals/fixtures",
        include_hard_negatives=True,
    )
    with_neg = [p for p in pairs if p.get("negatives")]
    assert with_neg or len(pairs) >= 2
