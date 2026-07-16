"""Komut satırı ingest / rebuild / eval / judge yardımcıları.

Örnekler:
  python -m rag.cli rebuild
  python -m rag.cli ingest path/to/file.pdf
  python -m rag.cli list
  python -m rag.cli eval
  python -m rag.cli judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from app.config import (
    DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DOCSTORE_PATH,
    INDEX_PATH,
    INDEXES_DIR,
    METADATA_DIR,
    MULTILINGUAL_EMBEDDING_MODEL,
)
from rag.embed import Embedder, resolve_embedding_model
from rag.eval import run_regression
from rag.hash_embed import HashEmbedder
from rag.index import FaissIndex
from rag.ingest import ingest_path, list_data_files, rebuild_from_data_dir
from rag.judge import run_judge_file

DEFAULT_EVAL_FIXTURES = os.path.join("evals", "fixtures")
DEFAULT_EVAL_CASES = os.path.join("evals", "cases.json")
DEFAULT_JUDGE_CASES = os.path.join("evals", "judge_cases.json")


def _ensure_dirs():
    for d in [DATA_DIR, INDEXES_DIR, METADATA_DIR]:
        os.makedirs(d, exist_ok=True)


def _load_index(dim: int, embedding_model: str) -> FaissIndex:
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCSTORE_PATH):
        try:
            idx = FaissIndex.load(INDEX_PATH, DOCSTORE_PATH)
            if idx.dim == dim and (
                not idx.embedding_model or idx.embedding_model == embedding_model
            ):
                if not idx.embedding_model:
                    idx.embedding_model = embedding_model
                return idx
        except Exception as exc:
            print(f"Mevcut indeks yüklenemedi, yeni oluşturulacak: {exc}", file=sys.stderr)
    return FaissIndex(dim=dim, embedding_model=embedding_model)


def cmd_list(_: argparse.Namespace) -> int:
    files = list_data_files(DATA_DIR)
    if not files:
        print("(data/ boş)")
        return 0
    for path in files:
        print(os.path.relpath(path, DATA_DIR).replace("\\", "/"))
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    _ensure_dirs()
    model = resolve_embedding_model(args.embedding)
    print(f"Embedding: {model}")
    emb = Embedder(model_name=model)
    index, reports = rebuild_from_data_dir(DATA_DIR, emb)
    index.save(INDEX_PATH, DOCSTORE_PATH)
    ok = [r for r in reports if not r.get("skipped")]
    bad = [r for r in reports if r.get("skipped")]
    print(f"Rebuild tamam: {len(ok)} dosya, {index.size} chunk -> {INDEX_PATH}")
    for r in bad:
        print(f"ATLANDI {r['source_file']}: {r.get('reason')}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    _ensure_dirs()
    model = resolve_embedding_model(args.embedding)
    emb = Embedder(model_name=model)
    index = _load_index(emb.dim, emb.model_name)

    paths = list(args.paths)
    if args.from_data:
        paths.extend(list_data_files(DATA_DIR))

    if not paths:
        print("İşlenecek dosya yok.", file=sys.stderr)
        return 1

    for path in paths:
        if not os.path.isfile(path):
            print(f"Yok: {path}", file=sys.stderr)
            continue
        # data/ dışındaysa kopyalama yapmadan doğrudan ingest et
        try:
            report = ingest_path(
                path,
                index,
                emb,
                replace_existing=True,
                folder=getattr(args, "folder", None),
                tags=getattr(args, "tags", None),
            )
            print(
                f"{report['source_file']}: +{report['chunks_added']} "
                f"(silinen={report['chunks_removed']}"
                f", klasör={report.get('folder') or '(kök)'}"
                f", etiket={','.join(report.get('tags') or []) or '-'})"
                + (f" ATLANDI: {report['reason']}" if report.get("skipped") else "")
            )
        except Exception as exc:
            print(f"HATA {path}: {exc}", file=sys.stderr)
    index.save(INDEX_PATH, DOCSTORE_PATH)
    print(f"Kaydedildi: {INDEX_PATH} ({index.size} chunk)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Fixture dokümanları + cases.json ile retrieval regression."""
    fixture_dir = args.fixtures
    cases_path = args.cases
    if not os.path.isdir(fixture_dir):
        print(f"Fixture klasörü yok: {fixture_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(cases_path):
        print(f"Cases dosyası yok: {cases_path}", file=sys.stderr)
        return 2

    if args.embedding == "hash":
        emb = HashEmbedder(dim=args.hash_dim)
        print("Embedding: hash-embedder (model indirmez)")
    else:
        model = resolve_embedding_model(args.embedding)
        print(f"Embedding: {model}")
        emb = Embedder(model_name=model)

    report = run_regression(
        fixture_dir,
        cases_path,
        emb,
        top_k=args.top_k,
        threshold=args.threshold,
        use_hybrid=not args.no_hybrid,
        min_accuracy=args.min_accuracy,
    )
    summary = report["summary"]
    print(
        f"Eval: {summary['passed']}/{summary['total']} "
        f"(accuracy={summary['accuracy']:.2%}, min={summary['min_accuracy']:.2%})"
    )
    for r in report["results"]:
        mark = "GEÇTI" if r["passed"] else "KALDI"
        cid = r.get("case_id") or "-"
        print(f"  [{mark}] {cid}: {r['question']} — {r['reason']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Rapor yazıldı: {args.output}")

    return 0 if summary.get("ok") else 1


def cmd_judge(args: argparse.Namespace) -> int:
    """Yanıt kalitesi: heuristic veya LLM-as-judge."""
    cases_path = args.cases
    if not os.path.isfile(cases_path):
        print(f"Judge cases dosyası yok: {cases_path}", file=sys.stderr)
        return 2

    report = run_judge_file(
        cases_path,
        mode=args.mode,
        provider=args.provider,
        model_name=args.model,
        min_accuracy=args.min_accuracy,
    )
    summary = report["summary"]
    print(
        f"Judge ({summary.get('mode')}): {summary['passed']}/{summary['total']} "
        f"(accuracy={summary['accuracy']:.2%}, min={summary['min_accuracy']:.2%})"
    )
    for r in report["results"]:
        mark = "GEÇTI" if r["passed"] else "KALDI"
        cid = r.get("case_id") or "-"
        print(
            f"  [{mark}] {cid}: grounded={r['grounded']} score={r['score']:.2f} — {r['reason']}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Rapor yazıldı: {args.output}")

    return 0 if summary.get("ok") else 1


def _add_embedding_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Preset (mini-en / mini-multi), 'hash' veya model adı. "
            f"Multilingual varsayılan: {MULTILINGUAL_EMBEDDING_MODEL}"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag.cli", description="RAG ingest CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="data/ altındaki desteklenen dosyaları listele")
    p_list.set_defaults(func=cmd_list)

    p_rebuild = sub.add_parser("rebuild", help="data/ üzerinden indeksi sıfırdan kur")
    _add_embedding_arg(p_rebuild)
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_ingest = sub.add_parser("ingest", help="Dosya(lar)ı indekse ekle/yenile")
    _add_embedding_arg(p_ingest)
    p_ingest.add_argument("paths", nargs="*", help="PDF/TXT yolları")
    p_ingest.add_argument(
        "--from-data",
        action="store_true",
        help="data/ klasöründeki tüm desteklenen dosyaları da ekle",
    )
    p_ingest.add_argument("--folder", default=None, help="Kaynak klasör etiketi (ör. hukuk)")
    p_ingest.add_argument(
        "--tags",
        default=None,
        help="Virgülle ayrılmış etiketler (ör. sozlesme,2024)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser("eval", help="Fixture + cases ile retrieval regression")
    p_eval.add_argument("--fixtures", default=DEFAULT_EVAL_FIXTURES, help="Fixture klasörü")
    p_eval.add_argument("--cases", default=DEFAULT_EVAL_CASES, help="cases.json yolu")
    p_eval.add_argument(
        "--embedding",
        default="hash",
        help="hash (varsayılan, hızlı) | mini-en | mini-multi | model adı",
    )
    p_eval.add_argument("--hash-dim", type=int, default=64, help="Hash embedder boyutu")
    p_eval.add_argument("--top-k", type=int, default=4)
    p_eval.add_argument("--threshold", type=float, default=0.30)
    p_eval.add_argument("--min-accuracy", type=float, default=1.0)
    p_eval.add_argument("--no-hybrid", action="store_true", help="Yalnızca dense retrieval")
    p_eval.add_argument("--output", default=None, help="JSON rapor çıktı yolu")
    p_eval.set_defaults(func=cmd_eval)

    p_judge = sub.add_parser("judge", help="Yanıt kalitesi (heuristic / LLM-as-judge)")
    p_judge.add_argument("--cases", default=DEFAULT_JUDGE_CASES, help="judge_cases.json yolu")
    p_judge.add_argument(
        "--mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="heuristic: hızlı token overlap; llm: model çağrısı",
    )
    p_judge.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    p_judge.add_argument("--model", default="phi3:mini", help="LLM model adı (mode=llm)")
    p_judge.add_argument("--min-accuracy", type=float, default=1.0)
    p_judge.add_argument("--output", default=None, help="JSON rapor çıktı yolu")
    p_judge.set_defaults(func=cmd_judge)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
