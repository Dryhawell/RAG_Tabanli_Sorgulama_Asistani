"""Komut satırı ingest / rebuild / eval / judge yardımcıları.

Örnekler:
  python -m rag.cli rebuild
  python -m rag.cli ingest path/to/file.pdf
  python -m rag.cli list
  python -m rag.cli eval
  python -m rag.cli judge
  python -m rag.cli stats
  python -m rag.cli prometheus
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
    METRICS_PATH,
    MULTILINGUAL_EMBEDDING_MODEL,
    PROMETHEUS_ADDR,
    PROMETHEUS_PORT,
    EMBED_FINETUNE_OUTPUT_DIR,
    EMBED_PIPELINE_REPORT_PATH,
    COLLAB_WS_HOST,
    COLLAB_WS_PORT,
    DOMAIN_PAIRS_PATH,
    DOMAIN_COLLECT_MIN_GATE,
    CHAT_DIR,
    AUDIT_LOG_PATH,
    FEDERATED_POOL_PATH,
    FEDERATED_MIN_PER_TENANT,
    PRIVATE_FEDERATED_POOL_PATH,
    FEDERATED_DP_NOISE,
    FEDERATED_DP_CLIP,
    FEDERATED_SECRET,
    ENABLE_FEDERATED_PRIVACY,
    DP_TRAIN_OUTPUT_DIR,
    DP_TRAIN_NOISE,
    DP_TRAIN_MAX_GRAD_NORM,
    DP_TRAIN_DELTA,
    DP_TRAIN_USE_OPACUS,
)
from rag.embed import Embedder, resolve_embedding_model
from rag.eval import run_regression
from rag.hash_embed import HashEmbedder
from rag.ingest import ingest_path, list_data_files, rebuild_from_data_dir
from rag.judge import run_judge_file
from rag.metrics import summarize_metrics
from rag.prometheus_sink import ensure_prometheus_server, prometheus_available, render_prometheus
from rag.embed_finetune import (
    build_pairs_from_eval,
    compare_embedding_models,
    export_pairs_jsonl,
    load_pairs_jsonl,
    run_embed_pipeline,
    train_embedding_model,
)
from rag.domain_collect import collect_and_save
from rag.federated_pool import build_federated_pool, aggregate_federated_pairs
from rag.privacy_federated import build_private_federated_pool
from rag.dp_train import opacus_available, train_embedding_model_dp
from rag.collab_ws import ensure_collab_ws_server, run_collab_ws_server, websockets_available
from rag.store import create_index, load_index

DEFAULT_EVAL_FIXTURES = os.path.join("evals", "fixtures")
DEFAULT_EVAL_CASES = os.path.join("evals", "cases.json")
DEFAULT_JUDGE_CASES = os.path.join("evals", "judge_cases.json")
DEFAULT_EMBED_PAIRS = os.path.join("metadata", "embed_pairs.jsonl")


def _ensure_dirs():
    for d in [DATA_DIR, INDEXES_DIR, METADATA_DIR]:
        os.makedirs(d, exist_ok=True)


def _load_index(dim: int, embedding_model: str):
    try:
        idx = load_index(
            INDEX_PATH,
            DOCSTORE_PATH,
            dim=dim,
            embedding_model=embedding_model,
        )
        if idx.dim == dim and (
            not idx.embedding_model or idx.embedding_model == embedding_model
        ):
            if not idx.embedding_model:
                idx.embedding_model = embedding_model
            return idx
    except Exception as exc:
        print(f"Mevcut indeks yüklenemedi, yeni oluşturulacak: {exc}", file=sys.stderr)
    return create_index(dim=dim, embedding_model=embedding_model)


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


def cmd_collab_serve(args: argparse.Namespace) -> int:
    if not websockets_available():
        print("websockets kurulu değil. pip install websockets", file=sys.stderr)
        return 2
    host = args.host or COLLAB_WS_HOST
    port = args.port or COLLAB_WS_PORT
    try:
        run_collab_ws_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\nKapatıldı.")
    return 0


def cmd_embed_pairs(args: argparse.Namespace) -> int:
    fixture_dir = args.fixtures
    cases_path = args.cases
    if not os.path.isdir(fixture_dir):
        print(f"Fixture klasörü yok: {fixture_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(cases_path):
        print(f"Cases dosyası yok: {cases_path}", file=sys.stderr)
        return 2
    pairs = build_pairs_from_eval(
        cases_path,
        fixture_dir,
        include_hard_negatives=args.hard_negatives,
    )
    if not pairs:
        print("Üretilen çift yok.", file=sys.stderr)
        return 1
    out = export_pairs_jsonl(pairs, args.output)
    print(f"{len(pairs)} çift yazıldı: {out}")
    return 0


def cmd_embed_train(args: argparse.Namespace) -> int:
    pairs_path = args.pairs
    if not os.path.isfile(pairs_path):
        print(f"Pairs dosyası yok: {pairs_path}", file=sys.stderr)
        return 2
    pairs = load_pairs_jsonl(pairs_path)
    if not pairs:
        print("Pairs dosyası boş.", file=sys.stderr)
        return 1
    out_dir = args.output or EMBED_FINETUNE_OUTPUT_DIR
    print(
        f"Eğitim: base={args.embedding} pairs={len(pairs)} "
        f"epochs={args.epochs} -> {out_dir}"
    )
    try:
        saved = train_embedding_model(
            args.embedding,
            pairs_path,
            out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"Model kaydedildi: {saved}")
        return 0
    except Exception as exc:
        print(f"Eğitim hatası: {exc}", file=sys.stderr)
        return 1


def cmd_embed_eval(args: argparse.Namespace) -> int:
    fixture_dir = args.fixtures
    cases_path = args.cases
    if not os.path.isdir(fixture_dir):
        print(f"Fixture klasörü yok: {fixture_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(cases_path):
        print(f"Cases dosyası yok: {cases_path}", file=sys.stderr)
        return 2
    if not args.finetuned:
        print("--finetuned gerekli", file=sys.stderr)
        return 2
    try:
        report = compare_embedding_models(
            args.embedding,
            args.finetuned,
            fixture_dir,
            cases_path,
            top_k=args.top_k,
            threshold=args.threshold,
            use_hybrid=not args.no_hybrid,
            min_accuracy=args.min_accuracy,
        )
    except Exception as exc:
        print(f"Karşılaştırma hatası: {exc}", file=sys.stderr)
        return 1
    b = report["base_summary"]
    f = report["finetuned_summary"]
    print(f"Base ({report['base_model']}): {b['passed']}/{b['total']} ({b['accuracy']:.2%})")
    print(
        f"Fine-tuned ({report['finetuned_model']}): "
        f"{f['passed']}/{f['total']} ({f['accuracy']:.2%})"
    )
    print(f"Delta accuracy: {report['delta_accuracy']:+.2%}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
        print(f"Rapor: {args.output}")
    return 0 if report.get("improved") else 1


def cmd_embed_pipeline(args: argparse.Namespace) -> int:
    fixture_dir = args.fixtures
    cases_path = args.cases
    if not os.path.isdir(fixture_dir):
        print(f"Fixture klasörü yok: {fixture_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(cases_path):
        print(f"Cases dosyası yok: {cases_path}", file=sys.stderr)
        return 2
    pairs_path = args.pairs or DEFAULT_EMBED_PAIRS
    out_dir = args.output or EMBED_FINETUNE_OUTPUT_DIR
    report_path = args.report or EMBED_PIPELINE_REPORT_PATH
    try:
        report = run_embed_pipeline(
            fixture_dir=fixture_dir,
            cases_path=cases_path,
            pairs_path=pairs_path,
            output_dir=out_dir,
            report_path=report_path,
            embedding=args.embedding,
            finetuned_dir=args.finetuned_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            top_k=args.top_k,
            threshold=args.threshold,
            use_hybrid=not args.no_hybrid,
            min_accuracy=args.min_accuracy,
            train=not args.pairs_only,
            hard_negatives=args.hard_negatives,
            collected_pairs_path=args.collected_pairs,
            federated_pool_path=args.federated_pool,
            private_pool_path=getattr(args, "private_pool", None),
        )
    except Exception as exc:
        print(f"Pipeline hatası: {exc}", file=sys.stderr)
        return 1
    print(
        f"Pipeline: pairs={report['pairs_count']} train={not args.pairs_only} "
        f"ok={report.get('ok')}"
    )
    if report.get("compare"):
        c = report["compare"]
        print(f"Delta accuracy: {c.get('delta_accuracy', 0):+.2%}")
    print(f"Rapor: {report_path}")
    return 0 if report.get("ok") else 1


def cmd_domain_collect(args: argparse.Namespace) -> int:
    chat_root = args.chats or CHAT_DIR
    audit_path = args.audit or AUDIT_LOG_PATH
    metrics_path = args.metrics or METRICS_PATH
    output = args.output or DOMAIN_PAIRS_PATH
    summary = collect_and_save(
        output,
        chat_root=chat_root if args.from_chats else None,
        audit_path=audit_path if args.from_audit else None,
        metrics_path=metrics_path if args.from_metrics else None,
        min_gate_score=args.min_gate,
    )
    print(
        f"Domain toplama: collected={summary['collected']} "
        f"added={summary['added']} total={summary['total']} -> {summary['output']}"
    )
    return 0


def cmd_federated_pool(args: argparse.Namespace) -> int:
    output = args.output or FEDERATED_POOL_PATH
    metadata_root = args.metadata or METADATA_DIR
    summary = build_federated_pool(
        output,
        metadata_root=metadata_root,
        min_per_tenant=args.min_per_tenant,
    )
    print(
        f"Federated havuz: total={summary['total']} tenants={summary['tenant_count']} "
        f"-> {summary['output']}"
    )
    if summary.get("tenants"):
        for tid, count in sorted(summary["tenants"].items()):
            print(f"  {tid}: {count}")
    return 0


def cmd_privacy_pool(args: argparse.Namespace) -> int:
    metadata_root = args.metadata or METADATA_DIR
    pairs = aggregate_federated_pairs(
        metadata_root,
        min_per_tenant=args.min_per_tenant,
    )
    if not pairs:
        print("Federated çift bulunamadı.", file=sys.stderr)
        return 1
    output = args.output or PRIVATE_FEDERATED_POOL_PATH
    report = build_private_federated_pool(
        pairs,
        output,
        noise_multiplier=args.noise,
        clip_norm=args.clip,
        shared_secret=args.secret or FEDERATED_SECRET,
        keep_text_pairs=not args.redact_text,
    )
    print(
        f"Private federated: pairs={report['pairs']} contributions={report['contributions']} "
        f"noise={report['noise_multiplier']} -> {report['output']}"
    )
    if report.get("tenants"):
        print("  tenants:", ", ".join(report["tenants"]))
    return 0


def cmd_dp_train(args: argparse.Namespace) -> int:
    pairs_path = args.pairs or DEFAULT_EMBED_PAIRS
    if not os.path.isfile(pairs_path):
        # eval'den üret
        from rag.embed_finetune import build_pairs_from_eval, export_pairs_jsonl

        pairs = build_pairs_from_eval(DEFAULT_EVAL_CASES, DEFAULT_EVAL_FIXTURES)
        export_pairs_jsonl(pairs, pairs_path)
        print(f"Pairs üretildi: {len(pairs)} -> {pairs_path}")
    out_dir = args.output or DP_TRAIN_OUTPUT_DIR
    use_opacus = args.opacus if args.opacus is not None else DP_TRAIN_USE_OPACUS
    if use_opacus and not opacus_available():
        print("Opacus yok; manuel DP-SGD fallback kullanılacak.", file=sys.stderr)
        use_opacus = False
    try:
        report = train_embedding_model_dp(
            args.embedding,
            pairs_path,
            out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_multiplier=args.noise,
            max_grad_norm=args.clip,
            delta=args.delta,
            use_opacus=use_opacus,
        )
    except Exception as exc:
        print(f"DP eğitim hatası: {exc}", file=sys.stderr)
        return 1
    print(
        f"DP-SGD: steps={report['steps']} ε≈{report['epsilon']:.3f} "
        f"opacus={report['used_opacus']} -> {report['weights']}"
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """JSONL metrik özetini yazdırır."""
    path = args.path or METRICS_PATH
    summary = summarize_metrics(path=path, limit=args.limit)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    q = summary["query"]
    print(f"Metrik özeti ({path})")
    print(f"  Toplam olay: {summary['total_events']}")
    print(f"  Sorgu: {q['count']} (no-answer={q['no_answer_rate']:.0%})")
    print(f"  Ort. gate={q['avg_gate_score']:.3f}  Ort. gecikme={q['avg_latency_ms']:.0f} ms")
    print(
        f"  Ingest: {summary['ingest']['count']} "
        f"({summary['ingest']['chunks_added']} chunk)  "
        f"Rebuild: {summary['rebuild']['count']}  "
        f"Silme: {summary['delete']['count']}"
    )
    if summary["by_kind"]:
        print("  Tür dağılımı:", ", ".join(f"{k}={v}" for k, v in sorted(summary["by_kind"].items())))
    return 0


def cmd_prometheus(args: argparse.Namespace) -> int:
    """Prometheus /metrics HTTP sunucusunu başlatır veya metrik dump alır."""
    if not prometheus_available():
        print(
            "prometheus_client kurulu değil. pip install prometheus-client",
            file=sys.stderr,
        )
        return 2

    if args.dump:
        sys.stdout.buffer.write(render_prometheus())
        return 0

    port = args.port or PROMETHEUS_PORT
    addr = args.addr or PROMETHEUS_ADDR
    ok = ensure_prometheus_server(port=port, addr=addr, enabled=True)
    if not ok:
        print("Prometheus sunucusu başlatılamadı", file=sys.stderr)
        return 1
    print(f"Prometheus /metrics dinleniyor: http://{addr}:{port}/metrics")
    print("Durdurmak için Ctrl+C")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nKapatıldı.")
    return 0


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

    p_stats = sub.add_parser("stats", help="Metrik özeti (JSONL)")
    p_stats.add_argument("--path", default=None, help=f"metrics.jsonl yolu (varsayılan: {METRICS_PATH})")
    p_stats.add_argument("--limit", type=int, default=5000, help="Okunacak son kayıt sayısı")
    p_stats.add_argument("--json", action="store_true", help="JSON çıktı")
    p_stats.set_defaults(func=cmd_stats)

    p_prom = sub.add_parser("prometheus", help="Prometheus /metrics sunucusu veya dump")
    p_prom.add_argument("--port", type=int, default=None, help=f"Port (varsayılan {PROMETHEUS_PORT})")
    p_prom.add_argument("--addr", default=None, help=f"Bind adresi (varsayılan {PROMETHEUS_ADDR})")
    p_prom.add_argument(
        "--dump",
        action="store_true",
        help="Sunucu başlatmadan metrikleri stdout'a yaz",
    )
    p_prom.set_defaults(func=cmd_prometheus)

    p_collab = sub.add_parser("collab-serve", help="İşbirlikçi not WebSocket sunucusu")
    p_collab.add_argument("--host", default=None, help=f"Bind host (varsayılan {COLLAB_WS_HOST})")
    p_collab.add_argument("--port", type=int, default=None, help=f"Port (varsayılan {COLLAB_WS_PORT})")
    p_collab.set_defaults(func=cmd_collab_serve)

    p_pairs = sub.add_parser("embed-pairs", help="Eval'den embedding eğitim çiftleri üret")
    p_pairs.add_argument("--fixtures", default=DEFAULT_EVAL_FIXTURES)
    p_pairs.add_argument("--cases", default=DEFAULT_EVAL_CASES)
    p_pairs.add_argument("--output", default=DEFAULT_EMBED_PAIRS)
    p_pairs.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Her çifte opsiyonel hard negative ekle",
    )
    p_pairs.set_defaults(func=cmd_embed_pairs)

    p_train = sub.add_parser("embed-train", help="Embedding contrastive fine-tuning")
    _add_embedding_arg(p_train)
    p_train.add_argument("--pairs", default=DEFAULT_EMBED_PAIRS, help="JSONL çift dosyası")
    p_train.add_argument("--output", default=None, help="Model çıktı dizini")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.set_defaults(func=cmd_embed_train)

    p_emeval = sub.add_parser("embed-eval", help="Base vs fine-tuned retrieval karşılaştır")
    _add_embedding_arg(p_emeval)
    p_emeval.add_argument("--fixtures", default=DEFAULT_EVAL_FIXTURES)
    p_emeval.add_argument("--cases", default=DEFAULT_EVAL_CASES)
    p_emeval.add_argument("--finetuned", required=True, help="Fine-tuned model dizini veya adı")
    p_emeval.add_argument("--top-k", type=int, default=4)
    p_emeval.add_argument("--threshold", type=float, default=0.30)
    p_emeval.add_argument("--min-accuracy", type=float, default=0.0)
    p_emeval.add_argument("--no-hybrid", action="store_true")
    p_emeval.add_argument("--output", default=None, help="JSON rapor")
    p_emeval.set_defaults(func=cmd_embed_eval)

    p_pipe = sub.add_parser("embed-pipeline", help="Pairs + train + eval tam döngü")
    _add_embedding_arg(p_pipe)
    p_pipe.add_argument("--fixtures", default=DEFAULT_EVAL_FIXTURES)
    p_pipe.add_argument("--cases", default=DEFAULT_EVAL_CASES)
    p_pipe.add_argument("--pairs", default=None, help="JSONL çift çıktısı")
    p_pipe.add_argument("--output", default=None, help="Model çıktı dizini")
    p_pipe.add_argument("--finetuned-dir", default=None, help="Eğitilmiş model dizini (override)")
    p_pipe.add_argument("--report", default=None, help="JSON pipeline raporu")
    p_pipe.add_argument("--epochs", type=int, default=1)
    p_pipe.add_argument("--batch-size", type=int, default=8)
    p_pipe.add_argument("--top-k", type=int, default=4)
    p_pipe.add_argument("--threshold", type=float, default=0.30)
    p_pipe.add_argument("--min-accuracy", type=float, default=0.0)
    p_pipe.add_argument("--no-hybrid", action="store_true")
    p_pipe.add_argument(
        "--pairs-only",
        action="store_true",
        help="Yalnızca çift üret (eğitim/eval atla)",
    )
    p_pipe.add_argument("--hard-negatives", action="store_true")
    p_pipe.add_argument(
        "--collected-pairs",
        default=None,
        help="Domain toplama JSONL (eval çiftleriyle birleştirilir)",
    )
    p_pipe.add_argument(
        "--federated-pool",
        default=None,
        help="Federated tenant havuzu JSONL",
    )
    p_pipe.add_argument(
        "--private-pool",
        default=None,
        help="DP/private federated pool JSONL",
    )
    p_pipe.set_defaults(func=cmd_embed_pipeline)

    p_fed = sub.add_parser("federated-pool", help="Tenant domain çiftlerini federated havuzda birleştir")
    p_fed.add_argument("--output", default=None, help="Havuz JSONL çıktısı")
    p_fed.add_argument("--metadata", default=None, help="metadata kök dizini")
    p_fed.add_argument("--min-per-tenant", type=int, default=FEDERATED_MIN_PER_TENANT)
    p_fed.set_defaults(func=cmd_federated_pool)

    p_priv = sub.add_parser(
        "privacy-pool",
        help="DP-SGD lite + secure aggregation ile private federated havuz",
    )
    p_priv.add_argument("--output", default=None, help="Private havuz çıktısı")
    p_priv.add_argument("--metadata", default=None)
    p_priv.add_argument("--min-per-tenant", type=int, default=FEDERATED_MIN_PER_TENANT)
    p_priv.add_argument("--noise", type=float, default=FEDERATED_DP_NOISE)
    p_priv.add_argument("--clip", type=float, default=FEDERATED_DP_CLIP)
    p_priv.add_argument("--secret", default=None, help="Secure aggregation secret")
    p_priv.add_argument(
        "--redact-text",
        action="store_true",
        help="Metin çiftlerini yazma (yalnızca DP aggregate meta)",
    )
    p_priv.set_defaults(func=cmd_privacy_pool)

    p_dp = sub.add_parser("dp-train", help="Opacus / DP-SGD ile embedding eğitimi")
    _add_embedding_arg(p_dp)
    p_dp.add_argument("--pairs", default=None, help="JSONL çift dosyası")
    p_dp.add_argument("--output", default=None, help="Model çıktı dizini")
    p_dp.add_argument("--epochs", type=int, default=2)
    p_dp.add_argument("--batch-size", type=int, default=4)
    p_dp.add_argument("--noise", type=float, default=DP_TRAIN_NOISE)
    p_dp.add_argument("--clip", type=float, default=DP_TRAIN_MAX_GRAD_NORM)
    p_dp.add_argument("--delta", type=float, default=DP_TRAIN_DELTA)
    p_dp.add_argument(
        "--opacus",
        dest="opacus",
        action="store_true",
        default=None,
        help="Opacus PrivacyEngine kullan",
    )
    p_dp.add_argument(
        "--no-opacus",
        dest="opacus",
        action="store_false",
        help="Manuel DP-SGD fallback",
    )
    p_dp.set_defaults(func=cmd_dp_train)

    p_dom = sub.add_parser("domain-collect", help="Sohbet/audit/metrikten domain çiftleri topla")
    p_dom.add_argument("--output", default=None, help="JSONL çıktı yolu")
    p_dom.add_argument("--chats", default=None, help="Sohbet kök dizini")
    p_dom.add_argument("--audit", default=None, help="audit.jsonl yolu")
    p_dom.add_argument("--metrics", default=None, help="metrics.jsonl yolu")
    p_dom.add_argument("--min-gate", type=float, default=DOMAIN_COLLECT_MIN_GATE)
    p_dom.add_argument("--from-chats", action="store_true", default=True)
    p_dom.add_argument("--no-chats", dest="from_chats", action="store_false")
    p_dom.add_argument("--from-audit", action="store_true", default=True)
    p_dom.add_argument("--no-audit", dest="from_audit", action="store_false")
    p_dom.add_argument("--from-metrics", action="store_true", default=True)
    p_dom.add_argument("--no-metrics", dest="from_metrics", action="store_false")
    p_dom.set_defaults(func=cmd_domain_collect)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
