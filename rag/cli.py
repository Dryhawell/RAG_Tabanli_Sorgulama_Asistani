"""Komut satırı ingest / rebuild yardımcıları.

Örnekler:
  python -m rag.cli rebuild
  python -m rag.cli ingest path/to/file.pdf
  python -m rag.cli list
"""

from __future__ import annotations

import argparse
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
from rag.index import FaissIndex
from rag.ingest import ingest_path, list_data_files, rebuild_from_data_dir


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
        print(os.path.basename(path))
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
            report = ingest_path(path, index, emb, replace_existing=True)
            print(
                f"{report['source_file']}: +{report['chunks_added']} "
                f"(silinen={report['chunks_removed']})"
                + (f" ATLANDI: {report['reason']}" if report.get("skipped") else "")
            )
        except Exception as exc:
            print(f"HATA {path}: {exc}", file=sys.stderr)
    index.save(INDEX_PATH, DOCSTORE_PATH)
    print(f"Kaydedildi: {INDEX_PATH} ({index.size} chunk)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag.cli", description="RAG ingest CLI")
    p.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Preset (mini-en / mini-multi) veya model adı. "
            f"Multilingual varsayılan: {MULTILINGUAL_EMBEDDING_MODEL}"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="data/ altındaki desteklenen dosyaları listele")
    p_list.set_defaults(func=cmd_list)

    p_rebuild = sub.add_parser("rebuild", help="data/ üzerinden indeksi sıfırdan kur")
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_ingest = sub.add_parser("ingest", help="Dosya(lar)ı indekse ekle/yenile")
    p_ingest.add_argument("paths", nargs="*", help="PDF/TXT yolları")
    p_ingest.add_argument(
        "--from-data",
        action="store_true",
        help="data/ klasöründeki tüm desteklenen dosyaları da ekle",
    )
    p_ingest.set_defaults(func=cmd_ingest)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
