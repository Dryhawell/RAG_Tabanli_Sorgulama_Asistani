"""Komut satırı ingest / rebuild / eval / judge yardımcıları.

Örnekler:
  python -m rag.cli rebuild
  python -m rag.cli rebuild --delta
  python -m rag.cli migrate-vector --source faiss --target qdrant
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
    COLLAB_HTTP_PORT,
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
    ST_DP_TRAIN_OUTPUT_DIR,
    ST_DP_HEAD_DIM,
    ST_DP_MAX_SEQ_LENGTH,
    ST_DP_FREEZE_BACKBONE,
    LORA_DP_TRAIN_OUTPUT_DIR,
    LORA_DP_RANK,
    LORA_DP_ALPHA,
    LORA_DP_MOCK,
    LORA_DP_OPACUS_PRODUCTION,
    LORA_DP_SECURE_MODE,
    LORA_DP_GRAD_SAMPLE_MODE,
    LORA_DP_EVAL_MIN_ACCURACY,
    LORA_DP_EVAL_MIN_DELTA,
    LORA_DP_EVAL_AUTO_ROLLBACK,
    JUDGE_MIN_ACCURACY,
    JUDGE_MODE,
)
from rag.embed import Embedder, resolve_embedding_model
from rag.eval import run_regression
from rag.hash_embed import HashEmbedder
from rag.ingest import (
    ingest_path,
    list_data_files,
    rebuild_delta_from_data_dir,
    rebuild_from_data_dir,
)
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
from rag.st_dp_train import train_sentence_transformer_dp_from_pairs_file
from rag.st_lora_dp import peft_available, train_lora_dp_from_pairs_file
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
    if getattr(args, "delta", False):
        index = _load_index(emb.dim, emb.model_name)
        index, reports, summary = rebuild_delta_from_data_dir(DATA_DIR, emb, index)
        index.save(INDEX_PATH, DOCSTORE_PATH)
        print(
            f"Delta rebuild: updated={summary.get('updated')} "
            f"unchanged={summary.get('unchanged')} removed={summary.get('removed')} "
            f"chunks={index.size} -> {INDEX_PATH}"
        )
        for r in reports:
            action = r.get("action") or ("skip" if r.get("skipped") else "ok")
            if action in {"unchanged"}:
                continue
            if r.get("skipped") and action != "removed":
                print(f"ATLANDI {r['source_file']}: {r.get('reason')}")
            elif action == "removed":
                print(f"SİLİNDİ {r['source_file']}: {r.get('chunks_removed', 0)} chunk")
            else:
                print(
                    f"{action.upper()} {r['source_file']}: "
                    f"+{r.get('chunks_added', 0)} -{r.get('chunks_removed', 0)}"
                )
        return 0
    index, reports = rebuild_from_data_dir(DATA_DIR, emb)
    index.save(INDEX_PATH, DOCSTORE_PATH)
    ok = [r for r in reports if not r.get("skipped")]
    bad = [r for r in reports if r.get("skipped")]
    print(f"Rebuild tamam: {len(ok)} dosya, {index.size} chunk -> {INDEX_PATH}")
    for r in bad:
        print(f"ATLANDI {r['source_file']}: {r.get('reason')}")
    return 0


def cmd_migrate_vector(args: argparse.Namespace) -> int:
    from rag.store import (
        DualWriteIndex,
        create_index,
        dual_write_backend,
        dual_write_catch_up,
        dual_write_lag_report,
        load_index,
        migrate_vector_store,
        report_dual_write_lag,
        write_cutover_env,
    )

    if getattr(args, "lag_report", False):
        index = load_index(INDEX_PATH, DOCSTORE_PATH)
        report = report_dual_write_lag(index)
        if not report.get("dual_write"):
            secondary = (getattr(args, "target", None) or dual_write_backend() or "qdrant")
            try:
                if not isinstance(index, DualWriteIndex) and secondary:
                    sec = create_index(
                        dim=getattr(index, "dim", 384),
                        embedding_model=getattr(index, "embedding_model", None),
                        backend=secondary,
                        dual_write="",
                    )
                    wrapped = DualWriteIndex(index, sec, secondary_backend=secondary)
                    report = dual_write_lag_report(wrapped)
            except Exception as exc:
                report = {"dual_write": False, "ok": False, "error": str(exc)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok", False) or report.get("reason") == "not_dual_write" else 1

    if getattr(args, "shadow_compare", False):
        from rag.store import dual_write_catch_up, dual_write_shadow_compare_from_index

        index = load_index(INDEX_PATH, DOCSTORE_PATH)
        if not isinstance(index, DualWriteIndex):
            secondary = (getattr(args, "target", None) or dual_write_backend() or "qdrant")
            try:
                sec = create_index(
                    dim=getattr(index, "dim", 384),
                    embedding_model=getattr(index, "embedding_model", None),
                    backend=secondary,
                    dual_write="",
                )
                index = DualWriteIndex(index, sec, secondary_backend=secondary)
            except Exception as exc:
                print(
                    json.dumps(
                        {"ok": False, "error": str(exc), "dual_write": False},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 1
        report = dual_write_shadow_compare_from_index(
            index,
            sample=int(getattr(args, "sample", 16) or 16),
            top_k=int(getattr(args, "top_k", 6) or 6),
            seed=int(getattr(args, "seed", 0) or 0),
        )
        if (
            not report.get("ok")
            and getattr(args, "auto_catch_up_on_fail", False)
            and isinstance(index, DualWriteIndex)
        ):
            catch_up = dual_write_catch_up(index)
            retry = dual_write_shadow_compare_from_index(
                index,
                sample=int(getattr(args, "sample", 16) or 16),
                top_k=int(getattr(args, "top_k", 6) or 6),
                seed=int(getattr(args, "seed", 0) or 0),
            )
            combined = {
                "ok": bool(retry.get("ok")),
                "auto_catch_up": True,
                "initial": report,
                "catch_up": catch_up,
                "retry": retry,
            }
            print(json.dumps(combined, ensure_ascii=False, indent=2, default=str))
            return 0 if combined.get("ok") else 1
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if getattr(args, "catch_up", False) and not getattr(args, "cutover", False):
        index = load_index(INDEX_PATH, DOCSTORE_PATH)
        if not isinstance(index, DualWriteIndex):
            print(
                json.dumps(
                    {"ok": False, "error": "not_dual_write"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        report = dual_write_catch_up(index)
        if getattr(args, "auto_cutover", False):
            lag = report.get("lag_after") or {}
            if (
                report.get("ok")
                and lag.get("ok")
                and int(lag.get("lag") or 0) == 0
            ):
                target = (args.target or "qdrant").strip().lower()
                out = args.write_env or os.path.join(
                    METADATA_DIR, f"vector_cutover_{target}.env"
                )
                path = write_cutover_env(
                    target_backend=target, path=out, clear_dual_write=True
                )
                report["auto_cutover"] = True
                report["cutover_env"] = path
                report["target"] = target
            else:
                report["auto_cutover"] = False
                report["auto_cutover_reason"] = "lag_not_zero_or_catch_up_failed"
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if getattr(args, "cutover", False):
        target = (args.target or "qdrant").strip().lower()
        out = args.write_env or os.path.join(METADATA_DIR, f"vector_cutover_{target}.env")
        index = load_index(INDEX_PATH, DOCSTORE_PATH)
        catch_up_report = None
        if (
            getattr(args, "catch_up", False)
            and isinstance(index, DualWriteIndex)
        ):
            catch_up_report = dual_write_catch_up(index)
        lag = dual_write_lag_report(index)
        if (
            lag.get("dual_write")
            and not lag.get("ok")
            and not getattr(args, "force", False)
        ):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "lag_not_zero",
                        "lag": lag,
                        "catch_up": catch_up_report,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 1
        path = write_cutover_env(target_backend=target, path=out, clear_dual_write=True)
        print(
            json.dumps(
                {
                    "ok": True,
                    "cutover_env": path,
                    "target": target,
                    "lag": lag,
                    "catch_up": catch_up_report,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0

    report = migrate_vector_store(
        source_backend=args.source,
        target_backend=args.target,
        index_path=INDEX_PATH,
        docstore_path=DOCSTORE_PATH,
        verify=not bool(args.no_verify),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


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
    http_port = args.http_port if args.http_port is not None else COLLAB_HTTP_PORT
    try:
        run_collab_ws_server(
            host=host,
            port=port,
            http_port=http_port,
            enable_http=not bool(args.no_http),
        )
    except KeyboardInterrupt:
        print("\nKapatıldı.")
    return 0


def cmd_collab_notifications(args: argparse.Namespace) -> int:
    from rag.collab_notify import (
        list_notifications_global,
        mark_notifications_read_global,
    )

    user = args.user or "local"
    if args.test_dispatch:
        from rag.collab_notify_dispatch import dispatch_notification

        sample = {
            "id": "test",
            "workspace_key": "test",
            "target_user": user,
            "from_user": "system",
            "body_preview": "Test bildirimi",
            "kind": "mention",
        }
        result = dispatch_notification(sample)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.register_push_token:
        from rag.collab_notify_push import register_device_token

        rec = register_device_token(
            user,
            args.register_push_token,
            platform=args.push_platform or "fcm",
            label=args.push_label,
            device_name=args.push_device_name,
            os_name=args.push_os_name,
            os_version=args.push_os_version,
            app_version=args.push_app_version,
            quiet_hours=args.push_quiet_hours,
            geofence_lat=args.push_geofence_lat,
            geofence_lon=args.push_geofence_lon,
            geofence_radius_m=args.push_geofence_radius,
            last_lat=args.push_last_lat,
            last_lon=args.push_last_lon,
            ttl_days=args.push_ttl_days,
        )
        print(json.dumps(rec, ensure_ascii=False))
        return 0
    if args.list_push_devices:
        from rag.collab_notify_push import summarize_user_devices

        devices = summarize_user_devices(user)
        print(json.dumps(devices, ensure_ascii=False, indent=2))
        return 0
    if args.revoke_push_token:
        from rag.collab_notify_push import revoke_device_token

        ok = revoke_device_token(user, args.revoke_push_token)
        print(json.dumps({"revoked": ok}, ensure_ascii=False))
        return 0 if ok else 1
    if args.prune_push_tokens:
        from rag.collab_notify_push import prune_expired_tokens

        result = prune_expired_tokens(username=user if not args.prune_all_users else None)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.test_push:
        from rag.collab_notify_push import dispatch_push

        ok = dispatch_push(
            user,
            title="RAG Collab test",
            body="Test mobil push bildirimi",
            data={"type": "test"},
        )
        print(json.dumps({"sent": ok}, ensure_ascii=False))
        return 0 if ok else 1
    if args.digest_report:
        from rag.collab_notify_digest import (
            export_digest_report_csv,
            summarize_digest_report,
        )

        if args.digest_report_csv:
            csv_text = export_digest_report_csv(
                tenant_id=args.digest_tenant,
                username=args.digest_report_user,
                since=args.digest_since,
                until=args.digest_until,
                limit=args.digest_report_limit,
            )
            out = args.digest_report_csv
            if out in {"-", "/dev/stdout"}:
                print(csv_text, end="")
            else:
                import os

                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                with open(out, "w", encoding="utf-8") as f:
                    f.write(csv_text)
                print(json.dumps({"wrote": out, "bytes": len(csv_text)}, ensure_ascii=False))
            return 0
        summary = summarize_digest_report(
            tenant_id=args.digest_tenant,
            username=args.digest_report_user,
            since=args.digest_since,
            until=args.digest_until,
            limit=args.digest_report_limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.digest_alert_check:
        from rag.collab_notify_digest import check_digest_alerts, check_digest_alerts_all

        if getattr(args, "digest_alert_all", False):
            result = check_digest_alerts_all(
                since=args.digest_since,
                until=args.digest_until,
                limit=args.digest_report_limit,
                dry_run=bool(args.digest_alert_dry_run),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result.get("any_failed_dispatch") else 0
        result = check_digest_alerts(
            tenant_id=args.digest_tenant,
            username=args.digest_report_user,
            since=args.digest_since,
            until=args.digest_until,
            limit=args.digest_report_limit,
            dry_run=bool(args.digest_alert_dry_run),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if (not result.get("fired")) or result.get("dispatched") or result.get("dry_run") else 1
    if args.generate_vapid:
        from rag.collab_notify_push import format_vapid_env, generate_vapid_keys

        keys = generate_vapid_keys(subject=args.vapid_subject)
        if args.vapid_write:
            import os

            path = args.vapid_write
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n" + format_vapid_env(keys))
            print(json.dumps({"wrote": path, "public": keys["public"]}, ensure_ascii=False))
        else:
            print(json.dumps({k: keys[k] for k in ("public", "subject")}, ensure_ascii=False, indent=2))
            print(format_vapid_env(keys), end="")
        return 0
    if args.rotate_vapid:
        from rag.collab_notify_push import format_vapid_env, rotate_vapid_keys

        result = rotate_vapid_keys(
            subject=args.vapid_subject,
            path=args.vapid_vault,
            archive_private=bool(args.vapid_archive_private),
            write_env=args.vapid_write,
        )
        out = {
            "public": result.get("public"),
            "subject": result.get("subject"),
            "fingerprint": result.get("fingerprint"),
            "archived_fingerprint": result.get("archived_fingerprint"),
            "path": result.get("path"),
            "history_len": result.get("history_len"),
        }
        if args.vapid_show_private:
            out["private"] = result.get("private")
            out["env"] = result.get("env")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if args.vapid_show_private:
            print(format_vapid_env(result), end="")
        if getattr(args, "update_github_secrets", False):
            from scripts.update_github_vapid_secrets import update_vapid_github_secrets

            gh = update_vapid_github_secrets(
                public=str(result.get("public") or ""),
                private=str(result.get("private") or "") or None,
                subject=str(result.get("subject") or "") or None,
                dry_run=bool(getattr(args, "vapid_secrets_dry_run", False)),
                include_private=bool(getattr(args, "vapid_secrets_include_private", False)),
                environment=(getattr(args, "vapid_github_environment", None) or None),
            )
            print(json.dumps({"github_secrets": gh}, ensure_ascii=False, indent=2))
            if gh.get("failed") and not gh.get("dry_run"):
                return 1
        return 0
    if args.webpush_sw:
        from rag.collab_notify_push import (
            load_service_worker_js,
            service_worker_js_path,
            vapid_application_server_key,
            vapid_public_key,
        )

        print(
            json.dumps(
                {
                    "sw_path": service_worker_js_path(),
                    "vapid_public": vapid_public_key() or None,
                    "application_server_key": vapid_application_server_key(),
                    "sw_js": load_service_worker_js()[:200] + "…",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.digest:
        from rag.collab_notify_digest import send_digest_email

        result = send_digest_email(
            user,
            hours=args.digest_hours,
            mentions_only=args.digest_mentions_only,
            group_by=args.digest_group_by,
            min_per_workspace=args.digest_min_per_workspace,
            ignore_quiet_hours=args.digest_force,
            quiet_hours=args.digest_quiet_hours,
            timezone_name=args.digest_timezone,
            tenant_id=args.digest_tenant,
        )
        print(json.dumps(result, ensure_ascii=False))
        empty_reasons = {
            "empty",
            "empty_after_filter",
            "below_workspace_threshold",
            "quiet_hours",
        }
        return 0 if result.get("sent") or result.get("reason") in empty_reasons else 1
    if args.digest_flush_quiet:
        from rag.collab_notify_digest import flush_quiet_hours_digests

        result = flush_quiet_hours_digests(
            hours=args.digest_hours,
            mentions_only=args.digest_mentions_only,
            group_by=args.digest_group_by,
            min_per_workspace=args.digest_min_per_workspace,
            quiet_hours=args.digest_quiet_hours,
            timezone_name=args.digest_timezone,
            tenant_id=args.digest_tenant,
            force=bool(args.digest_force),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.digest_all:
        from rag.collab_notify_digest import send_digest_all

        result = send_digest_all(
            hours=args.digest_hours,
            mentions_only=args.digest_mentions_only,
            group_by=args.digest_group_by,
            min_per_workspace=args.digest_min_per_workspace,
            ignore_quiet_hours=args.digest_force,
            quiet_hours=args.digest_quiet_hours,
            timezone_name=args.digest_timezone,
            tenant_id=args.digest_tenant,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.mark_read:
        changed = mark_notifications_read_global(
            user,
            notification_ids=args.ids if args.ids else None,
        )
        print(f"Okundu işaretlendi: {changed}")
        return 0
    rows = list_notifications_global(
        user,
        limit=args.limit,
        unread_only=args.unread_only,
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print(f"Bildirim yok ({user})")
        return 0
    for row in rows:
        read = "✓" if row.get("read") else "•"
        print(
            f"{read} [{row.get('workspace_key')}] "
            f"{row.get('from_user') or '-'}: {row.get('body_preview') or ''}"
        )
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


def cmd_st_dp_train(args: argparse.Namespace) -> int:
    pairs_path = args.pairs or DEFAULT_EMBED_PAIRS
    if not os.path.isfile(pairs_path):
        pairs = build_pairs_from_eval(DEFAULT_EVAL_CASES, DEFAULT_EVAL_FIXTURES)
        export_pairs_jsonl(pairs, pairs_path)
        print(f"Pairs üretildi: {len(pairs)} -> {pairs_path}")
    out_dir = args.output or ST_DP_TRAIN_OUTPUT_DIR
    use_opacus = args.opacus if args.opacus is not None else DP_TRAIN_USE_OPACUS
    if use_opacus and not opacus_available():
        print("Opacus yok; manuel DP-SGD fallback kullanılacak.", file=sys.stderr)
        use_opacus = False
    try:
        report = train_sentence_transformer_dp_from_pairs_file(
            args.embedding,
            pairs_path,
            out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_multiplier=args.noise,
            max_grad_norm=args.clip,
            delta=args.delta,
            use_opacus=use_opacus,
            head_dim=args.head_dim,
            max_seq_length=args.max_seq_length,
            freeze_backbone=not args.unfreeze_backbone,
        )
    except Exception as exc:
        print(f"ST+DP eğitim hatası: {exc}", file=sys.stderr)
        return 1
    print(
        f"ST+DP: steps={report['steps']} ε≈{report['epsilon']:.3f} "
        f"opacus={report['used_opacus']} head={report['weights']}"
    )
    return 0


def cmd_lora_dp_train(args: argparse.Namespace) -> int:
    pairs_path = args.pairs or DEFAULT_EMBED_PAIRS
    if not os.path.isfile(pairs_path):
        pairs = build_pairs_from_eval(DEFAULT_EVAL_CASES, DEFAULT_EVAL_FIXTURES)
        export_pairs_jsonl(pairs, pairs_path)
        print(f"Pairs üretildi: {len(pairs)} -> {pairs_path}")
    out_dir = args.output or LORA_DP_TRAIN_OUTPUT_DIR
    use_opacus = args.opacus if args.opacus is not None else DP_TRAIN_USE_OPACUS
    if use_opacus and not opacus_available():
        print("Opacus yok; manuel DP-SGD fallback kullanılacak.", file=sys.stderr)
        use_opacus = False
    mock = bool(args.mock) if args.mock is not None else LORA_DP_MOCK
    production = bool(args.production) if args.production is not None else LORA_DP_OPACUS_PRODUCTION
    if not mock and not peft_available():
        print("PEFT yok; mock LoRA yoluna düşülüyor.", file=sys.stderr)
        mock = True
    try:
        extra: dict = {}
        if not mock:
            extra["lora_alpha"] = args.alpha
            extra["production_mode"] = production
            extra["secure_mode"] = (
                bool(args.secure_mode) if args.secure_mode is not None else LORA_DP_SECURE_MODE
            )
            extra["grad_sample_mode"] = args.grad_sample_mode or LORA_DP_GRAD_SAMPLE_MODE
        report = train_lora_dp_from_pairs_file(
            args.embedding,
            pairs_path,
            out_dir,
            mock=mock,
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_multiplier=args.noise,
            max_grad_norm=args.clip,
            delta=args.delta,
            use_opacus=use_opacus,
            lora_rank=args.rank,
            **extra,
        )
    except Exception as exc:
        print(f"LoRA+DP eğitim hatası: {exc}", file=sys.stderr)
        return 1
    print(
        f"LoRA+DP: steps={report['steps']} ε≈{report['epsilon']:.3f} "
        f"opacus={report['used_opacus']} peft={report.get('used_peft')} "
        f"format={report.get('format')}"
    )
    return 0


def cmd_lora_dp_eval(args: argparse.Namespace) -> int:
    from rag.lora_dp_eval import (
        check_lora_delta_gate,
        check_lora_eval_gate,
        check_lora_eval_gates,
        lora_adapter_available,
        run_lora_dp_eval_report,
    )

    lora_dir = args.lora_dir or LORA_DP_TRAIN_OUTPUT_DIR
    if not lora_adapter_available(lora_dir):
        print(f"LoRA adapter bulunamadı: {lora_dir}/lora_adapter", file=sys.stderr)
        return 2
    min_acc = (
        args.min_accuracy
        if args.min_accuracy is not None
        else LORA_DP_EVAL_MIN_ACCURACY
    )
    min_delta = (
        args.min_delta
        if args.min_delta is not None
        else LORA_DP_EVAL_MIN_DELTA
    )
    try:
        report = run_lora_dp_eval_report(
            args.embedding,
            lora_dir,
            args.fixtures or DEFAULT_EVAL_FIXTURES,
            args.cases or DEFAULT_EVAL_CASES,
            report_path=args.output,
            per_case_path=args.per_case_output,
            rollback_path=args.rollback_output,
            auto_rollback=bool(args.auto_rollback or LORA_DP_EVAL_AUTO_ROLLBACK),
            apply_env_patch=args.apply_env_patch,
            rebuild_dry_run=args.rebuild_dry_run,
            rebuild_confirm=args.confirm_rebuild,
            env_path=args.env_patch_output,
            data_dir=args.data_dir,
            top_k=args.top_k,
            threshold=args.threshold,
            use_hybrid=not args.no_hybrid,
            min_accuracy=min_acc,
            min_delta=min_delta,
        )
    except Exception as exc:
        print(f"LoRA+DP eval hatası: {exc}", file=sys.stderr)
        return 1
    b = report["base_summary"]["accuracy"]
    l = report["lora_summary"]["accuracy"]
    d = report["delta_accuracy"]
    print(
        f"LoRA eval: base={b:.2%} lora={l:.2%} delta={d:+.2%} "
        f"improved={report['improved']} min_acc={min_acc:.2%} "
        f"lora_ok={report.get('lora_ok')} min_delta={min_delta:+.2%} "
        f"delta_ok={report.get('delta_ok')} regressions={report.get('regression_count', 0)}"
    )
    for reg in report.get("regressions") or []:
        cid = reg.get("case_id") or "-"
        print(
            f"  [REGRESS] {cid}: base_gate={reg.get('base_gate', 0):.3f} "
            f"lora_gate={reg.get('lora_gate', 0):.3f} "
            f"Δgate={reg.get('gate_delta', 0):+.3f}",
            file=sys.stderr,
        )
    suggestion = report.get("rollback_suggestion") or {}
    if suggestion.get("should_rollback"):
        print(
            f"Rollback önerisi: reason={suggestion.get('reason')} "
            f"actions={len(suggestion.get('actions') or [])}",
            file=sys.stderr,
        )
        for act in suggestion.get("actions") or []:
            print(f"  → {act}", file=sys.stderr)
        if report.get("rollback_path"):
            print(f"Rollback JSON: {report['rollback_path']}", file=sys.stderr)
        exec_result = report.get("rollback_execution") or {}
        if exec_result.get("env", {}).get("applied"):
            print(
                f"Env patch yazıldı: {exec_result['env'].get('env_path')}",
                file=sys.stderr,
            )
        if exec_result.get("rebuild", {}).get("would_rebuild"):
            rb = exec_result["rebuild"]
            print(
                f"Rebuild dry-run: {rb.get('source_count', 0)} dosya — {rb.get('command')}",
                file=sys.stderr,
            )
    if not check_lora_eval_gates(report, min_acc, min_delta):
        if not check_lora_eval_gate(report, min_acc):
            print(
                f"LoRA accuracy gate başarısız: lora={l:.2%} < min={min_acc:.2%}",
                file=sys.stderr,
            )
        if not check_lora_delta_gate(report, min_delta):
            print(
                f"LoRA delta gate başarısız: delta={d:+.2%} < min_delta={min_delta:+.2%}",
                file=sys.stderr,
            )
        return 1
    if args.per_case_output:
        print(f"Per-case rapor: {args.per_case_output}")
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


def cmd_judge_ack_export(args: argparse.Namespace) -> int:
    from rag.judge_alert import export_judge_ack_audit

    report = export_judge_ack_audit(
        fmt=getattr(args, "format", None) or "jsonl",
        path=getattr(args, "audit", None),
        output=getattr(args, "output", None),
        limit=getattr(args, "limit", None),
        since=getattr(args, "since", None),
        until=getattr(args, "until", None),
        event=getattr(args, "event", None),
        include_state=bool(getattr(args, "include_state", False)),
        state_path=getattr(args, "state", None),
    )
    if report.get("output"):
        print(json.dumps({k: v for k, v in report.items() if k != "text"}, indent=2))
    else:
        sys.stdout.write(report.get("text") or "")
    return 0 if report.get("ok") else 1


def cmd_judge_ack_purge(args: argparse.Namespace) -> int:
    from rag.judge_alert import purge_judge_ack_audit

    report = purge_judge_ack_audit(
        path=getattr(args, "audit", None),
        days=getattr(args, "days", None),
        keep=getattr(args, "keep", None),
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


def cmd_judge_ack_digest(args: argparse.Namespace) -> int:
    from rag.judge_alert import (
        dispatch_judge_ack_digest,
        dispatch_judge_ack_digest_fanout,
        summarize_judge_ack_audit,
    )

    hours = float(getattr(args, "hours", 168) or 168)
    dry_run = bool(getattr(args, "dry_run", False))
    quiet = getattr(args, "quiet_hours", None)
    force = bool(getattr(args, "force", False))
    tz = getattr(args, "timezone", None)
    audit = getattr(args, "audit", None)
    block_kit = None
    if getattr(args, "no_block_kit", False):
        block_kit = False
    elif getattr(args, "block_kit", False):
        block_kit = True

    if getattr(args, "fan_out", False):
        report = dispatch_judge_ack_digest_fanout(
            path=audit,
            since_hours=hours,
            dry_run=dry_run,
            quiet_hours=quiet,
            ignore_quiet_hours=force,
            timezone_name=tz,
            webhook=getattr(args, "webhook", None),
            block_kit=block_kit,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    summary = summarize_judge_ack_audit(
        path=audit,
        since_hours=hours,
        tenant_id=getattr(args, "tenant", None),
    )
    dispatched = dispatch_judge_ack_digest(
        summary,
        webhook=getattr(args, "webhook", None),
        dry_run=dry_run,
        tenant_id=getattr(args, "tenant", None),
        quiet_hours=quiet,
        ignore_quiet_hours=force,
        timezone_name=tz,
        block_kit=block_kit,
    )
    print(
        json.dumps(
            {"summary": summary, "dispatch": dispatched},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary.get("ok") and dispatched.get("ok") else 1


def cmd_alertmanager(args: argparse.Namespace) -> int:
    from rag.alertmanager_ops import (
        apply_inhibit_equal_with_gate,
        check_alertmanager_config,
        create_silence,
        delete_silence,
        list_alerts,
        list_silences,
        parse_duration_sec,
        parse_silence_matcher,
        reload_alertmanager,
        render_alertmanager_config,
        rotate_alertmanager_slack_webhook,
        tune_inhibit_equal_from_live,
        write_inhibit_rules,
    )

    if getattr(args, "check_config", False):
        path = getattr(args, "output", None)
        report = check_alertmanager_config(path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "list_alerts", False):
        report = list_alerts(base_url=getattr(args, "api_url", None))
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if getattr(args, "tune_equal", False) and not getattr(args, "apply_equal", False):
        report = tune_inhibit_equal_from_live(
            base_url=getattr(args, "api_url", None),
            paths=getattr(args, "alerting_path", None) or None,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "apply_equal", False) or getattr(args, "diff_inhibit", False):
        equal = [
            x.strip()
            for x in str(getattr(args, "equal", None) or "alertname,service").split(",")
            if x.strip()
        ]
        # --diff-inhibit alone is dry-run; --apply-equal writes after gate
        dry = bool(getattr(args, "diff_inhibit", False)) and not bool(
            getattr(args, "apply_equal", False)
        )
        dry = dry or bool(getattr(args, "dry_run", False))
        report = apply_inhibit_equal_with_gate(
            paths=getattr(args, "alerting_path", None) or None,
            output=getattr(args, "output", None),
            from_live=True,
            api_url=getattr(args, "api_url", None),
            equal_labels=equal,
            require_amtool=bool(getattr(args, "require_amtool", False)),
            dry_run=dry,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report.get("ok") else 1

    if getattr(args, "generate_inhibit", False):
        equal = [
            x.strip()
            for x in str(getattr(args, "equal", None) or "alertname,service").split(",")
            if x.strip()
        ]
        paths = getattr(args, "alerting_path", None) or None
        report = write_inhibit_rules(
            paths=paths,
            output=getattr(args, "output", None),
            equal_labels=equal,
            from_live=bool(getattr(args, "from_live", False)),
            api_url=getattr(args, "api_url", None),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "silence", False):
        matchers = []
        for raw in getattr(args, "matcher", None) or []:
            try:
                matchers.append(parse_silence_matcher(raw))
            except ValueError as exc:
                print(
                    json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                    file=sys.stderr,
                )
                return 2
        duration = None
        if getattr(args, "duration", None):
            try:
                duration = parse_duration_sec(args.duration)
            except ValueError as exc:
                print(
                    json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
                    file=sys.stderr,
                )
                return 2
        report = create_silence(
            matchers=matchers,
            duration_sec=duration,
            ends_at=getattr(args, "ends_at", None),
            created_by=getattr(args, "created_by", None) or "rag-cli",
            comment=getattr(args, "comment", None) or "",
            base_url=getattr(args, "api_url", None),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "list_silences", False):
        report = list_silences(base_url=getattr(args, "api_url", None))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "delete_silence", None):
        report = delete_silence(
            args.delete_silence, base_url=getattr(args, "api_url", None)
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "rotate_slack_webhook", None):
        report = rotate_alertmanager_slack_webhook(
            args.rotate_slack_webhook,
            write_env=args.write_env,
            output=args.output,
            reload=not bool(args.no_reload),
            reload_url=args.reload_url,
            webhook_url=args.webhook_url,
        )
        if getattr(args, "update_github_secrets", False):
            from scripts.update_github_alertmanager_secrets import (
                update_alertmanager_github_secrets,
            )

            report["github_secrets"] = update_alertmanager_github_secrets(
                slack_webhook=args.rotate_slack_webhook,
                webhook_url=args.webhook_url,
                dry_run=bool(getattr(args, "secrets_dry_run", False)),
                environment=getattr(args, "github_environment", None) or None,
            )
            if report["github_secrets"].get("failed") and not report["github_secrets"].get(
                "dry_run"
            ):
                report["ok"] = False
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1

    if getattr(args, "render", False):
        report = render_alertmanager_config(
            output=args.output,
            slack_webhook=args.slack_webhook,
            webhook_url=args.webhook_url,
            inhibit_path=getattr(args, "inhibit", None),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report.get("ok"):
            return 1
        if getattr(args, "reload", False):
            reloaded = reload_alertmanager(url=args.reload_url)
            print(json.dumps({"reload": reloaded}, ensure_ascii=False, indent=2))
            return 0 if reloaded.get("ok") else 1
        return 0

    if getattr(args, "reload", False):
        reloaded = reload_alertmanager(url=args.reload_url)
        print(json.dumps(reloaded, ensure_ascii=False, indent=2))
        return 0 if reloaded.get("ok") else 1

    print(
        "Kullanım: alertmanager --render | --reload | --rotate-slack-webhook | --silence | --generate-inhibit [--from-live] | --tune-equal | --diff-inhibit | --apply-equal | --list-alerts | --check-config",
        file=sys.stderr,
    )
    return 2


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
    p_rebuild.add_argument(
        "--delta",
        action="store_true",
        help="Incremental rebuild: değişmeyen kaynakları atla (ingest_manifest.json)",
    )
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_mig = sub.add_parser(
        "migrate-vector",
        help="FAISS ↔ Qdrant indeks migrasyonu (vektör kopyala + doğrula)",
    )
    p_mig.add_argument("--source", default="faiss", help="Kaynak backend (faiss|qdrant)")
    p_mig.add_argument("--target", default="qdrant", help="Hedef backend (faiss|qdrant)")
    p_mig.add_argument(
        "--no-verify",
        action="store_true",
        help="Kaynak/hedef size+sources doğrulamasını atla",
    )
    p_mig.add_argument(
        "--lag-report",
        action="store_true",
        help="Dual-write lag raporu (primary vs secondary size/sources)",
    )
    p_mig.add_argument(
        "--shadow-compare",
        action="store_true",
        help="Dual-write shadow-read: primary vs secondary search overlap",
    )
    p_mig.add_argument(
        "--auto-catch-up-on-fail",
        action="store_true",
        help="Shadow-compare fail olursa catch-up + yeniden gate",
    )
    p_mig.add_argument(
        "--sample",
        type=int,
        default=16,
        help="Shadow-compare örnek vektör sayısı",
    )
    p_mig.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Shadow-compare top_k",
    )
    p_mig.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shadow-compare RNG seed",
    )
    p_mig.add_argument(
        "--catch-up",
        action="store_true",
        help="Dual-write: eksik kaynakları primary→secondary replay (cutover ile birlikte de çalışır)",
    )
    p_mig.add_argument(
        "--auto-cutover",
        action="store_true",
        help="Catch-up sonrası lag=0 ise cutover env yaz",
    )
    p_mig.add_argument(
        "--cutover",
        action="store_true",
        help="Hedef backend için cutover env dosyası yaz (lag=0 gerekir)",
    )
    p_mig.add_argument(
        "--force",
        action="store_true",
        help="Cutover'da lag kontrolünü atla",
    )
    p_mig.add_argument(
        "--write-env",
        default=None,
        help="Cutover env çıktı yolu (varsayılan metadata/vector_cutover_<target>.env)",
    )
    p_mig.set_defaults(func=cmd_migrate_vector)

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
        default=JUDGE_MODE if JUDGE_MODE in {"heuristic", "llm"} else "heuristic",
        help="heuristic: hızlı token overlap; llm: model çağrısı",
    )
    p_judge.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    p_judge.add_argument("--model", default="phi3:mini", help="LLM model adı (mode=llm)")
    p_judge.add_argument(
        "--min-accuracy",
        type=float,
        default=JUDGE_MIN_ACCURACY,
        help="CI gate eşiği (RAG_JUDGE_MIN_ACCURACY)",
    )
    p_judge.add_argument("--output", default=None, help="JSON rapor çıktı yolu")
    p_judge.set_defaults(func=cmd_judge)

    p_jexp = sub.add_parser(
        "judge-ack-export",
        help="Judge soft-fail ack audit trail export (JSONL/CSV)",
    )
    p_jexp.add_argument(
        "--format",
        choices=["jsonl", "csv"],
        default="jsonl",
        help="Çıktı formatı",
    )
    p_jexp.add_argument("--output", default=None, help="Dosyaya yaz (yoksa stdout)")
    p_jexp.add_argument("--audit", default=None, help="Audit JSONL yolu")
    p_jexp.add_argument("--state", default=None, help="Judge alert state yolu")
    p_jexp.add_argument("--limit", type=int, default=None, help="Son N kayıt")
    p_jexp.add_argument("--since", default=None, help="ISO başlangıç filtresi")
    p_jexp.add_argument("--until", default=None, help="ISO bitiş filtresi")
    p_jexp.add_argument("--event", default=None, help="Olay filtresi (ack/alert/resolve)")
    p_jexp.add_argument(
        "--include-state",
        action="store_true",
        help="Mevcut state snapshot satırını ekle",
    )
    p_jexp.set_defaults(func=cmd_judge_ack_export)

    p_jpurge = sub.add_parser(
        "judge-ack-purge",
        help="Judge ack audit retention purge (days/keep, dry-run)",
    )
    p_jpurge.add_argument("--audit", default=None, help="Audit JSONL yolu")
    p_jpurge.add_argument(
        "--days",
        type=float,
        default=None,
        help="Bu günden eski kayıtları sil",
    )
    p_jpurge.add_argument(
        "--keep",
        type=int,
        default=None,
        help="Son N kaydı tut (days sonrası uygulanır)",
    )
    p_jpurge.add_argument(
        "--dry-run",
        action="store_true",
        help="Silmeden before/after/removed raporu yaz",
    )
    p_jpurge.set_defaults(func=cmd_judge_ack_purge)

    p_jdig = sub.add_parser(
        "judge-ack-digest",
        help="Judge ack audit haftalık özet (Slack webhook)",
    )
    p_jdig.add_argument("--audit", default=None, help="Audit JSONL yolu")
    p_jdig.add_argument(
        "--hours",
        type=float,
        default=168.0,
        help="Özet penceresi (saat, varsayılan 168=7g)",
    )
    p_jdig.add_argument(
        "--webhook",
        default=None,
        help="Slack webhook (varsayılan RAG_JUDGE_SLACK_WEBHOOK)",
    )
    p_jdig.add_argument(
        "--tenant",
        default=None,
        help="Tek tenant özeti / webhook çözümleme",
    )
    p_jdig.add_argument(
        "--fan-out",
        action="store_true",
        help="RAG_JUDGE_ACK_DIGEST_WEBHOOKS_JSON tenant fan-out",
    )
    p_jdig.add_argument(
        "--quiet-hours",
        default=None,
        help="Quiet hours HH:MM-HH:MM (varsayılan RAG_JUDGE_ACK_DIGEST_QUIET_HOURS)",
    )
    p_jdig.add_argument(
        "--timezone",
        default=None,
        help="Quiet hours timezone (IANA)",
    )
    p_jdig.add_argument(
        "--force",
        action="store_true",
        help="Quiet hours'ı yok say",
    )
    p_jdig.add_argument(
        "--block-kit",
        action="store_true",
        help="Slack Block Kit payload zorla",
    )
    p_jdig.add_argument(
        "--no-block-kit",
        action="store_true",
        help="Sadece düz text payload",
    )
    p_jdig.add_argument(
        "--dry-run",
        action="store_true",
        help="Webhook göndermeden özet yazdır",
    )
    p_jdig.set_defaults(func=cmd_judge_ack_digest)

    p_stats = sub.add_parser("stats", help="Metrik özeti (JSONL)")
    p_stats.add_argument("--path", default=None, help=f"metrics.jsonl yolu (varsayılan: {METRICS_PATH})")
    p_stats.add_argument("--limit", type=int, default=5000, help="Okunacak son kayıt sayısı")
    p_stats.add_argument("--json", action="store_true", help="JSON çıktı")
    p_stats.set_defaults(func=cmd_stats)

    p_prom = sub.add_parser("prometheus", help="Prometheus /metrics sunucusu veya dump")
    p_prom.add_argument("--dump", action="store_true", help="Metrikleri stdout'a yaz")
    p_prom.add_argument("--port", type=int, default=None, help="Dinleme portu")
    p_prom.add_argument("--addr", default=None, help="Dinleme adresi")
    p_prom.set_defaults(func=cmd_prometheus)

    p_am = sub.add_parser(
        "alertmanager",
        help="Alertmanager config render / reload / Slack webhook rotate",
    )
    p_am.add_argument("--render", action="store_true", help="Template → YAML render")
    p_am.add_argument("--reload", action="store_true", help="POST /-/reload")
    p_am.add_argument(
        "--check-config",
        action="store_true",
        help="amtool check-config (veya yapısal fallback)",
    )
    p_am.add_argument(
        "--generate-inhibit",
        action="store_true",
        help="Grafana label'larından Alertmanager inhibit_rules üret",
    )
    p_am.add_argument(
        "--from-live",
        action="store_true",
        help="Inhibit equal label'larını canlı Alertmanager alert'lerinden ayarla",
    )
    p_am.add_argument(
        "--tune-equal",
        action="store_true",
        help="Canlı+statik label'lardan equal öner (yazmadan)",
    )
    p_am.add_argument(
        "--diff-inhibit",
        action="store_true",
        help="Equal tune adayını diff'le (dry-run, yazmadan)",
    )
    p_am.add_argument(
        "--apply-equal",
        action="store_true",
        help="Equal tune + amtool/structural gate sonrası inhibit dosyasına yaz",
    )
    p_am.add_argument(
        "--require-amtool",
        action="store_true",
        help="--apply-equal/--diff-inhibit için structural fallback'i reddet",
    )
    p_am.add_argument(
        "--dry-run",
        action="store_true",
        help="Apply/diff yollarında yazmadan çalış",
    )
    p_am.add_argument(
        "--list-alerts",
        action="store_true",
        help="Alertmanager /api/v2/alerts listele",
    )
    p_am.add_argument(
        "--equal",
        default="alertname,service",
        help="Inhibit equal label listesi (virgülle)",
    )
    p_am.add_argument(
        "--alerting-path",
        action="append",
        default=None,
        help="Grafana alert/rule YAML yolu (tekrarlanabilir; varsayılan grafana/alerting+rules)",
    )
    p_am.add_argument(
        "--silence",
        action="store_true",
        help="Alertmanager silence oluştur (POST /api/v2/silences)",
    )
    p_am.add_argument(
        "--matcher",
        action="append",
        default=[],
        help="Silence matcher: name=value veya name=~regex (tekrarlanabilir)",
    )
    p_am.add_argument(
        "--duration",
        default=None,
        help="Silence süresi (ör. 2h, 30m, 900)",
    )
    p_am.add_argument(
        "--ends-at",
        default=None,
        help="Silence bitiş zamanı (RFC3339; --duration yerine)",
    )
    p_am.add_argument(
        "--comment",
        default="",
        help="Silence açıklaması",
    )
    p_am.add_argument(
        "--created-by",
        default="rag-cli",
        help="Silence createdBy",
    )
    p_am.add_argument(
        "--list-silences",
        action="store_true",
        help="Aktif silences listele",
    )
    p_am.add_argument(
        "--delete-silence",
        default=None,
        metavar="ID",
        help="Silence sil",
    )
    p_am.add_argument(
        "--api-url",
        default=None,
        help="Alertmanager base URL (varsayılan http://127.0.0.1:9093)",
    )
    p_am.add_argument(
        "--rotate-slack-webhook",
        default=None,
        metavar="URL",
        help="Slack webhook yaz + render (+ reload)",
    )
    p_am.add_argument(
        "--no-reload",
        action="store_true",
        help="Rotate sırasında reload atla",
    )
    p_am.add_argument(
        "--reload-url",
        default=None,
        help="Alertmanager reload URL (varsayılan http://127.0.0.1:9093/-/reload)",
    )
    p_am.add_argument(
        "--slack-webhook",
        default=None,
        help="Render için RAG_ALERTMANAGER_SLACK_WEBHOOK",
    )
    p_am.add_argument(
        "--webhook-url",
        default=None,
        help="Render için RAG_ALERTMANAGER_WEBHOOK_URL",
    )
    p_am.add_argument(
        "--output",
        default=None,
        help="Rendered YAML yolu",
    )
    p_am.add_argument(
        "--inhibit",
        default=None,
        help="Inhibit rules YAML (render merge; varsayılan ALERTMANAGER_INHIBIT)",
    )
    p_am.add_argument(
        "--write-env",
        default=None,
        help="Rotate env çıktı yolu (varsayılan metadata/alertmanager.slack.env)",
    )
    p_am.add_argument(
        "--update-github-secrets",
        action="store_true",
        help="Rotate sonrası GitHub Actions secrets güncelle (GH_PAT)",
    )
    p_am.add_argument(
        "--secrets-dry-run",
        action="store_true",
        help="GitHub secret güncellemeyi dry-run yap",
    )
    p_am.add_argument(
        "--github-environment",
        default=None,
        help="GitHub Environment adı (boş = repo secrets)",
    )
    p_am.set_defaults(func=cmd_alertmanager)

    p_collab = sub.add_parser("collab-serve", help="İşbirlikçi not WebSocket sunucusu")
    p_collab.add_argument("--host", default=None, help=f"Bind host (varsayılan {COLLAB_WS_HOST})")
    p_collab.add_argument("--port", type=int, default=None, help=f"Port (varsayılan {COLLAB_WS_PORT})")
    p_collab.add_argument(
        "--http-port",
        type=int,
        default=None,
        help=f"Web Push HTTP port (varsayılan {COLLAB_HTTP_PORT})",
    )
    p_collab.add_argument(
        "--no-http",
        action="store_true",
        help="Collab HTTP (SW/webpush) sunucusunu başlatma",
    )
    p_collab.set_defaults(func=cmd_collab_serve)

    p_cnot = sub.add_parser("collab-notifications", help="Çapraz workspace bildirim merkezi")
    p_cnot.add_argument("--user", default="local", help="Hedef kullanıcı adı")
    p_cnot.add_argument("--limit", type=int, default=50)
    p_cnot.add_argument("--unread-only", action="store_true", default=False)
    p_cnot.add_argument("--mark-read", action="store_true", help="Tümünü okundu işaretle")
    p_cnot.add_argument("--test-dispatch", action="store_true", help="E-posta/webhook test gönderimi")
    p_cnot.add_argument("--digest", action="store_true", help="Günlük özet e-postası gönder")
    p_cnot.add_argument("--digest-all", action="store_true", help="Tüm hedef kullanıcılara digest")
    p_cnot.add_argument("--digest-hours", type=int, default=24, help="Özet penceresi (saat)")
    p_cnot.add_argument(
        "--digest-mentions-only",
        action="store_true",
        help="Digest yalnızca @mention bildirimleri",
    )
    p_cnot.add_argument(
        "--digest-group-by",
        default=None,
        choices=["thread", "workspace", "none"],
        help="Digest gruplama (thread / workspace / none)",
    )
    p_cnot.add_argument(
        "--digest-min-per-workspace",
        type=int,
        default=None,
        help="Workspace min bildirim eşiği (varsayılan RAG_NOTIFY_DIGEST_MIN_PER_WORKSPACE)",
    )
    p_cnot.add_argument(
        "--digest-quiet-hours",
        default=None,
        help="Quiet hours UTC/yerel HH:MM-HH:MM (ör. 22:00-07:00)",
    )
    p_cnot.add_argument(
        "--digest-timezone",
        default=None,
        help="IANA timezone (ör. Europe/Istanbul)",
    )
    p_cnot.add_argument(
        "--digest-tenant",
        default=None,
        help="Tenant id (yerel saat eşlemesi için)",
    )
    p_cnot.add_argument(
        "--digest-force",
        action="store_true",
        help="Quiet hours'ı yok sayarak digest gönder",
    )
    p_cnot.add_argument(
        "--digest-flush-quiet",
        action="store_true",
        help="Quiet hours bitince bekleyen tam digest flush",
    )
    p_cnot.add_argument(
        "--digest-report",
        action="store_true",
        help="Tenant digest rapor özetini göster (JSONL dashboard)",
    )
    p_cnot.add_argument(
        "--digest-report-limit",
        type=int,
        default=200,
        help="Digest rapor satır limiti",
    )
    p_cnot.add_argument(
        "--digest-report-csv",
        default=None,
        metavar="PATH",
        help="Digest raporunu CSV yaz (PATH veya -)",
    )
    p_cnot.add_argument(
        "--digest-report-user",
        default=None,
        help="Digest rapor kullanıcı filtresi",
    )
    p_cnot.add_argument(
        "--digest-since",
        default=None,
        help="Digest rapor başlangıç ISO zamanı",
    )
    p_cnot.add_argument(
        "--digest-until",
        default=None,
        help="Digest rapor bitiş ISO zamanı",
    )
    p_cnot.add_argument(
        "--webpush-sw",
        action="store_true",
        help="Web Push SW yolu ve VAPID public özeti",
    )
    p_cnot.add_argument(
        "--generate-vapid",
        action="store_true",
        help="Yeni VAPID anahtar çifti üret",
    )
    p_cnot.add_argument(
        "--rotate-vapid",
        action="store_true",
        help="VAPID rotate + vault history (private varsayılan yazılmaz)",
    )
    p_cnot.add_argument(
        "--vapid-vault",
        default=None,
        help="VAPID vault JSON yolu (varsayılan metadata/vapid_keys.json)",
    )
    p_cnot.add_argument(
        "--vapid-archive-private",
        action="store_true",
        help="Rotate sırasında private'ı vault'a yaz (önerilmez)",
    )
    p_cnot.add_argument(
        "--vapid-show-private",
        action="store_true",
        help="Rotate çıktısında private/env göster",
    )
    p_cnot.add_argument(
        "--update-github-secrets",
        action="store_true",
        help="Rotate sonrası VAPID GitHub Actions secrets güncelle (GH_PAT)",
    )
    p_cnot.add_argument(
        "--vapid-secrets-dry-run",
        action="store_true",
        help="GitHub secret güncellemeyi simüle et",
    )
    p_cnot.add_argument(
        "--vapid-secrets-include-private",
        action="store_true",
        help="PRIVATE secret'ı da güncelle (dikkat)",
    )
    p_cnot.add_argument(
        "--vapid-github-environment",
        "--github-environment",
        dest="vapid_github_environment",
        default=None,
        help="GitHub Environment adı (Actions Environments secret sync)",
    )
    p_cnot.add_argument(
        "--vapid-subject",
        default=None,
        help="VAPID subject (mailto:...)",
    )
    p_cnot.add_argument(
        "--vapid-write",
        default=None,
        metavar="PATH",
        help="VAPID env satırlarını dosyaya ekle",
    )
    p_cnot.add_argument(
        "--digest-alert-check",
        action="store_true",
        help="Digest skip/fail eşik alert kontrolü",
    )
    p_cnot.add_argument(
        "--digest-alert-dry-run",
        action="store_true",
        help="Alert kontrolü yap, webhook gönderme",
    )
    p_cnot.add_argument(
        "--digest-alert-all",
        action="store_true",
        help="Tüm tenant'lar için digest alert fan-out",
    )
    p_cnot.add_argument(
        "--register-push-token",
        default=None,
        help="Mobil cihaz token kaydet (FCM/APNs PoC)",
    )
    p_cnot.add_argument(
        "--push-platform",
        default="fcm",
        choices=["fcm", "apns", "generic", "webpush"],
        help="Push platform",
    )
    p_cnot.add_argument("--push-label", default=None, help="Cihaz etiketi")
    p_cnot.add_argument("--push-device-name", default=None, help="Cihaz adı")
    p_cnot.add_argument("--push-os-name", default=None, help="OS adı (iOS/Android)")
    p_cnot.add_argument("--push-os-version", default=None, help="OS sürümü")
    p_cnot.add_argument("--push-app-version", default=None, help="Uygulama sürümü")
    p_cnot.add_argument(
        "--push-quiet-hours",
        default=None,
        help="Cihaz quiet hours override (HH:MM-HH:MM veya off)",
    )
    p_cnot.add_argument("--push-geofence-lat", type=float, default=None, help="Geofence enlem")
    p_cnot.add_argument("--push-geofence-lon", type=float, default=None, help="Geofence boylam")
    p_cnot.add_argument(
        "--push-geofence-radius",
        type=float,
        default=None,
        help="Geofence yarıçap (metre)",
    )
    p_cnot.add_argument("--push-last-lat", type=float, default=None, help="Son konum enlem")
    p_cnot.add_argument("--push-last-lon", type=float, default=None, help="Son konum boylam")
    p_cnot.add_argument("--push-ttl-days", type=int, default=None, help="Token TTL (gün)")
    p_cnot.add_argument(
        "--list-push-devices",
        action="store_true",
        help="Kayıtlı cihazları listele",
    )
    p_cnot.add_argument(
        "--revoke-push-token",
        default=None,
        help="Belirli token'ı iptal et",
    )
    p_cnot.add_argument(
        "--prune-push-tokens",
        action="store_true",
        help="Süresi dolmuş token'ları temizle",
    )
    p_cnot.add_argument(
        "--prune-all-users",
        action="store_true",
        help="Tüm kullanıcıların süresi dolmuş token'larını temizle",
    )
    p_cnot.add_argument(
        "--test-push",
        action="store_true",
        help="Kayıtlı token'a test push gönder",
    )
    p_cnot.add_argument("--ids", nargs="*", default=None, help="Belirli bildirim id'leri")
    p_cnot.add_argument("--json", action="store_true", help="JSON çıktı")
    p_cnot.set_defaults(func=cmd_collab_notifications)

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

    p_stdp = sub.add_parser(
        "st-dp-train",
        help="SentenceTransformer + Opacus DP fine-tune (frozen backbone + head)",
    )
    _add_embedding_arg(p_stdp)
    p_stdp.add_argument("--pairs", default=None)
    p_stdp.add_argument("--output", default=None)
    p_stdp.add_argument("--epochs", type=int, default=1)
    p_stdp.add_argument("--batch-size", type=int, default=4)
    p_stdp.add_argument("--noise", type=float, default=DP_TRAIN_NOISE)
    p_stdp.add_argument("--clip", type=float, default=DP_TRAIN_MAX_GRAD_NORM)
    p_stdp.add_argument("--delta", type=float, default=DP_TRAIN_DELTA)
    p_stdp.add_argument("--head-dim", type=int, default=ST_DP_HEAD_DIM)
    p_stdp.add_argument("--max-seq-length", type=int, default=ST_DP_MAX_SEQ_LENGTH)
    p_stdp.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Omurgayı da eğit (Opacus ile ağır)",
    )
    p_stdp.add_argument("--opacus", dest="opacus", action="store_true", default=None)
    p_stdp.add_argument("--no-opacus", dest="opacus", action="store_false")
    p_stdp.set_defaults(func=cmd_st_dp_train)

    p_lora = sub.add_parser(
        "lora-dp-train",
        help="LoRA + Opacus DP-SGD (PEFT veya mock tiny transformer)",
    )
    _add_embedding_arg(p_lora)
    p_lora.add_argument("--pairs", default=None)
    p_lora.add_argument("--output", default=None)
    p_lora.add_argument("--epochs", type=int, default=1)
    p_lora.add_argument("--batch-size", type=int, default=4)
    p_lora.add_argument("--noise", type=float, default=DP_TRAIN_NOISE)
    p_lora.add_argument("--clip", type=float, default=DP_TRAIN_MAX_GRAD_NORM)
    p_lora.add_argument("--delta", type=float, default=DP_TRAIN_DELTA)
    p_lora.add_argument("--rank", type=int, default=LORA_DP_RANK)
    p_lora.add_argument("--alpha", type=int, default=LORA_DP_ALPHA)
    p_lora.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        default=None,
        help="Tiny transformer + manuel LoRA (CI)",
    )
    p_lora.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Gerçek SentenceTransformer + PEFT LoRA",
    )
    p_lora.add_argument("--opacus", dest="opacus", action="store_true", default=None)
    p_lora.add_argument("--no-opacus", dest="opacus", action="store_false")
    p_lora.add_argument(
        "--production",
        dest="production",
        action="store_true",
        default=None,
        help="Opacus ModuleValidator + üretim grad_sample",
    )
    p_lora.add_argument(
        "--no-production",
        dest="production",
        action="store_false",
        help="Üretim Opacus sarmalayıcısını kapat",
    )
    p_lora.add_argument(
        "--secure-mode",
        dest="secure_mode",
        action="store_true",
        default=None,
        help="Opacus secure RNG (üretim)",
    )
    p_lora.add_argument(
        "--no-secure-mode",
        dest="secure_mode",
        action="store_false",
        help="Secure RNG kapalı",
    )
    p_lora.add_argument(
        "--grad-sample-mode",
        default=None,
        help="Opacus grad_sample_mode (varsayılan hooks)",
    )
    p_lora.set_defaults(func=cmd_lora_dp_train)

    p_loraev = sub.add_parser(
        "lora-dp-eval",
        help="Base vs LoRA+DP adapter retrieval karşılaştırması",
    )
    _add_embedding_arg(p_loraev)
    p_loraev.add_argument("--lora-dir", default=None, help="LoRA eğitim çıktı dizini")
    p_loraev.add_argument("--fixtures", default=None)
    p_loraev.add_argument("--cases", default=None)
    p_loraev.add_argument("--top-k", type=int, default=4)
    p_loraev.add_argument("--threshold", type=float, default=0.30)
    p_loraev.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="LoRA accuracy eşiği (varsayılan RAG_LORA_DP_EVAL_MIN_ACCURACY)",
    )
    p_loraev.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Base'e göre min delta (varsayılan RAG_LORA_DP_EVAL_MIN_DELTA)",
    )
    p_loraev.add_argument("--no-hybrid", action="store_true")
    p_loraev.add_argument("--output", default=None, help="JSON rapor")
    p_loraev.add_argument(
        "--per-case-output",
        default=None,
        help="Case-level delta JSON raporu (per_case_deltas + regressions)",
    )
    p_loraev.add_argument(
        "--rollback-output",
        default=None,
        help="Rollback önerisi JSON yolu",
    )
    p_loraev.add_argument(
        "--auto-rollback",
        action="store_true",
        help="Gate/regression başarısızsa rollback_suggestion.json yaz",
    )
    p_loraev.add_argument(
        "--apply-env-patch",
        action="store_true",
        help="Rollback env_patch'i dosyaya yaz ve os.environ'a uygula",
    )
    p_loraev.add_argument(
        "--env-patch-output",
        default=None,
        help="Env patch çıktı yolu (varsayılan metadata/lora_rollback.env)",
    )
    p_loraev.add_argument(
        "--rebuild-dry-run",
        action="store_true",
        help="Rollback sonrası rebuild dry-run (indeks yazmaz)",
    )
    p_loraev.add_argument(
        "--confirm-rebuild",
        action="store_true",
        help="Onaylı gerçek rebuild (indeks yazar; dikkatli kullanın)",
    )
    p_loraev.add_argument(
        "--data-dir",
        default=None,
        help="Rebuild dry-run/confirm için data dizini",
    )
    p_loraev.set_defaults(func=cmd_lora_dp_eval)

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
