"""Eval fixture'larından embedding fine-tuning çiftleri ve eğitim pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.eval import load_cases, run_regression
from rag.embed import resolve_embedding_model


def _fixture_path(fixture_dir: str, expected_source: str) -> Optional[str]:
    direct = os.path.join(fixture_dir, expected_source)
    if os.path.isfile(direct):
        return direct
    for name in os.listdir(fixture_dir):
        if name == expected_source or os.path.basename(name) == expected_source:
            path = os.path.join(fixture_dir, name)
            if os.path.isfile(path):
                return path
    return None


def _passage_from_fixture(fixture_dir: str, expected_source: str, max_chars: int = 900) -> str:
    path = _fixture_path(fixture_dir, expected_source)
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return ""
    # İlk anlamlı paragraf veya üst sınır
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    passage = parts[0] if parts else text
    if len(passage) > max_chars:
        passage = passage[:max_chars].rstrip() + "…"
    return passage


def build_pairs_from_eval(
    cases_path: str,
    fixture_dir: str,
    *,
    include_hard_negatives: bool = False,
) -> List[Dict[str, Any]]:
    """Pozitif eval vakalarından (soru, passage) çiftleri üretir."""
    cases = load_cases(cases_path)
    pairs: List[Dict[str, Any]] = []
    positives_by_source: Dict[str, str] = {}

    for case in cases:
        if case.expect_no_answer or not case.expected_source:
            continue
        passage = _passage_from_fixture(fixture_dir, case.expected_source)
        if not passage:
            continue
        positives_by_source[case.expected_source] = passage
        item: Dict[str, Any] = {
            "anchor": case.question,
            "positive": passage,
            "case_id": case.id,
            "expected_source": case.expected_source,
        }
        if include_hard_negatives:
            for src, other in positives_by_source.items():
                if src != case.expected_source:
                    item.setdefault("negatives", []).append(other)
                    break
        pairs.append(item)
    return pairs


def export_pairs_jsonl(pairs: Sequence[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_pairs_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("anchor") and row.get("positive"):
                rows.append(row)
    return rows


def train_embedding_model(
    base_model: str,
    pairs_path: str,
    output_dir: str,
    *,
    epochs: int = 1,
    batch_size: int = 8,
) -> str:
    """Sentence-transformers ile contrastive fine-tuning (MultipleNegativesRankingLoss)."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    pairs = load_pairs_jsonl(pairs_path)
    if not pairs:
        raise ValueError("Eğitim çifti yok")

    model_name = resolve_embedding_model(base_model)
    model = SentenceTransformer(model_name)
    examples = [
        InputExample(texts=[p["anchor"], p["positive"]]) for p in pairs
    ]
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=max(1, batch_size))
    train_loss = losses.MultipleNegativesRankingLoss(model)
    os.makedirs(output_dir, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=max(1, epochs),
        warmup_steps=0,
        output_path=output_dir,
        show_progress_bar=True,
    )
    return output_dir


def compare_embedding_models(
    base_model: str,
    finetuned_model: str,
    fixture_dir: str,
    cases_path: str,
    *,
    top_k: int = 4,
    threshold: float = 0.30,
    use_hybrid: bool = True,
    min_accuracy: float = 0.0,
) -> Dict[str, Any]:
    """Base vs fine-tuned retrieval accuracy karşılaştırması."""
    from rag.embed import Embedder

    base_name = resolve_embedding_model(base_model)
    ft_name = finetuned_model if os.path.isdir(finetuned_model) else resolve_embedding_model(
        finetuned_model
    )

    emb_base = Embedder(model_name=base_name)
    emb_ft = Embedder(model_name=ft_name)

    report_base = run_regression(
        fixture_dir,
        cases_path,
        emb_base,
        top_k=top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        min_accuracy=min_accuracy,
    )
    report_ft = run_regression(
        fixture_dir,
        cases_path,
        emb_ft,
        top_k=top_k,
        threshold=threshold,
        use_hybrid=use_hybrid,
        min_accuracy=min_accuracy,
    )

    b_acc = report_base["summary"]["accuracy"]
    f_acc = report_ft["summary"]["accuracy"]
    return {
        "base_model": base_name,
        "finetuned_model": ft_name,
        "base_summary": report_base["summary"],
        "finetuned_summary": report_ft["summary"],
        "delta_accuracy": f_acc - b_acc,
        "improved": f_acc >= b_acc,
    }


def run_embed_pipeline(
    *,
    fixture_dir: str,
    cases_path: str,
    pairs_path: str,
    output_dir: str,
    report_path: Optional[str] = None,
    embedding: str = "mini-en",
    finetuned_dir: Optional[str] = None,
    epochs: int = 1,
    batch_size: int = 8,
    top_k: int = 4,
    threshold: float = 0.30,
    use_hybrid: bool = True,
    min_accuracy: float = 0.0,
    train: bool = True,
    hard_negatives: bool = False,
) -> Dict[str, Any]:
    """Pairs → train → eval tek adımda; CI ve periyodik döngü için."""
    from datetime import datetime, timezone

    pairs = build_pairs_from_eval(
        cases_path,
        fixture_dir,
        include_hard_negatives=hard_negatives,
    )
    if not pairs:
        raise ValueError("Eğitim çifti üretilemedi")

    export_pairs_jsonl(pairs, pairs_path)
    out_dir = finetuned_dir or output_dir
    compare: Optional[Dict[str, Any]] = None
    trained_path: Optional[str] = None

    if train:
        trained_path = train_embedding_model(
            embedding,
            pairs_path,
            out_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
        compare = compare_embedding_models(
            embedding,
            trained_path,
            fixture_dir,
            cases_path,
            top_k=top_k,
            threshold=threshold,
            use_hybrid=use_hybrid,
            min_accuracy=min_accuracy,
        )

    report: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pairs_path": pairs_path,
        "pairs_count": len(pairs),
        "embedding": embedding,
        "output_dir": out_dir,
        "trained_path": trained_path,
        "epochs": epochs if train else 0,
        "compare": compare,
        "ok": True if not train else bool(compare and compare.get("improved")),
    }
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def pipeline_report_dict(
    pairs: Sequence[Dict[str, Any]],
    compare: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "pairs_count": len(pairs),
        "pairs_preview": list(pairs)[:3],
        "compare": compare,
    }
