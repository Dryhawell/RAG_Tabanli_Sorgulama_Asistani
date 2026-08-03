import os
import re
import json
import secrets
import time
import streamlit as st
import requests

from rag.embed import (
    EMBEDDING_PRESETS,
    Embedder,
    default_embedding_preset,
    preset_for_model,
    resolve_embedding_model,
)
from rag.hybrid import build_bm25_from_index
from rag.ingest import (
    delete_source,
    ensure_data_path,
    ingest_path,
    list_data_files,
    rebuild_from_data_dir,
)
from rag.i18n import SUPPORTED_LANGS, get_language, set_language, t
from rag.store import create_index, load_index, vector_backend
from rag.meta_store import normalize_folder, normalize_tags
from rag.eval import EvalCase, evaluate_cases, summarize
from rag.prompt import build_prompt
from rag.llm import generate_answer, stream_answer
from rag.query_rewrite import embed_rewrite, rewrite_query
from rag.compare import compare_sources, summarize_sources
from rag.agent import run_agent_loop, run_heuristic_tools, tools_context_block
from rag.highlight import highlight_answer_html
from rag.memory import (
    load_memory_from_session,
    save_memory_to_session,
    update_memory_after_turn,
)
from rag.planner import run_planned_agent
from rag.vision import (
    image_query_context,
    is_table_question,
    merge_image_into_question,
    prioritize_table_chunks,
)
from rag.profile_memory import (
    add_memory,
    ingest_session_facts,
    load_profile_memory,
    memories_context_block,
    search_memories,
)
from rag.rerank import get_reranker
from rag.retrieve import retrieve
from rag.readers import _ocr_available
from rag.chat_store import (
    append_messages,
    create_session,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)
from rag.auth import (
    add_user,
    authenticate,
    auth_enabled,
    delete_user,
    ensure_users_file,
    list_users_detail,
    update_user_acl,
    upsert_oidc_user,
)
from rag.acl import (
    can_ingest_to,
    filter_folder_options,
    filter_tag_options,
    intersect_folder_filter,
    intersect_tag_filter,
)
from rag.export_chat import export_session_json, export_session_markdown
from rag.share_links import (
    build_share_url,
    create_share_link,
    list_links_for_session,
    load_shared_session,
    resolve_share_token,
    revoke_share_link,
)
from rag.collab_notes import load_note, save_note
from rag.collab_crdt import load_crdt
from rag.collab_undo import apply_text_edit_with_undo, redo_edit, undo_edit
from rag.collab_component import render_collab_live_editor
from rag.collab_richtext import (
    add_comment,
    add_mark,
    load_richtext,
    marks_in_range,
    render_rich_html,
    render_mention_html,
    reply_comment,
    resolve_comment,
    summarize_mark_layers,
)
from rag.collab_richtext_audit import read_mark_audit, read_comment_audit
from rag.collab_richtext_diff import summarize_mark_audit_diffs, summarize_comment_audit_diffs
from rag.collab_notify import list_notifications, mark_notifications_read, notify_mentions
from rag.collab_notify import list_notifications_global, mark_notifications_read_global
from rag.collab_ws import ensure_collab_ws_server, websockets_available
from rag.audit import read_audit, write_audit
from rag.metrics import record_metric, summarize_metrics
from rag.oidc import (
    OIDCError,
    build_authorize_url,
    complete_login,
    fetch_discovery,
    generate_pkce_pair,
    load_oidc_config,
    oidc_enabled,
)
from rag.workspace import ensure_workspace_dirs, resolve_workspace
from app.config import (
    DATA_DIR,
    INDEXES_DIR,
    METADATA_DIR,
    CHAT_DIR,
    ENABLE_AUTH,
    AUTH_SHARED_INDEX,
    ENABLE_TENANTS,
    DEFAULT_TENANT,
    ENABLE_METRICS,
    ENABLE_PROMETHEUS,
    PROMETHEUS_PORT,
    OIDC_ONLY,
    DEFAULT_UI_LANG,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    NO_ANSWER_THRESHOLD,
    HYBRID_ALPHA,
    ENABLE_RERANKER,
    DEFAULT_RERANKER_MODEL,
    ENABLE_OCR,
    ENABLE_LAYOUT_PDF,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
    ENABLE_QUERY_REWRITE,
    DEFAULT_QUERY_REWRITE_MODE,
    ENABLE_AGENT_TOOLS,
    AGENT_MAX_STEPS,
    ENABLE_SOURCE_HIGHLIGHT,
    HIGHLIGHT_MIN_TOKENS,
    ENABLE_AGENT_MEMORY,
    ENABLE_AGENT_PLANNER,
    ENABLE_TABLE_BOOST,
    ENABLE_IMAGE_OCR,
    ENABLE_VISION_LLM,
    DEFAULT_VISION_OPENAI_MODEL,
    DEFAULT_VISION_OLLAMA_MODEL,
    ENABLE_PROFILE_MEMORY,
    PROFILE_MEMORY_TOP_K,
    PROFILE_MEMORY_MIN_SCORE,
    ENABLE_SHARE_LINKS,
    SHARE_LINK_DEFAULT_TTL_DAYS,
    PUBLIC_BASE_URL,
    ENABLE_COLLAB_NOTES,
    ENABLE_COLLAB_WS,
    COLLAB_WS_HOST,
    COLLAB_WS_PORT,
    COLLAB_WS_PUBLIC_HOST,
    ENABLE_COLLAB_CRDT,
    ENABLE_COLLAB_LIVE_EDITOR,
    ENABLE_COLLAB_RICHTEXT,
    ENABLE_DOMAIN_COLLECT,
    DOMAIN_PAIRS_PATH,
    DOMAIN_COLLECT_MIN_GATE,
)

st.set_page_config(page_title="RAG Not/PDF Asistanı", layout="wide")

if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = DEFAULT_UI_LANG if DEFAULT_UI_LANG in SUPPORTED_LANGS else "tr"
set_language(st.session_state["ui_lang"])
st.title(t("app_title"))

# Opsiyonel Prometheus scrape endpoint (UI process içinde)
if ENABLE_PROMETHEUS:
    try:
        from rag.prometheus_sink import ensure_prometheus_server

        ensure_prometheus_server()
    except Exception:
        pass

# Opsiyonel işbirlikçi WebSocket sunucusu (UI process içinde)
if ENABLE_COLLAB_WS and ENABLE_COLLAB_NOTES:
    try:
        ensure_collab_ws_server(
            host=COLLAB_WS_HOST,
            port=COLLAB_WS_PORT,
            enabled=True,
        )
    except Exception:
        pass

for _d in [DATA_DIR, INDEXES_DIR, METADATA_DIR, CHAT_DIR]:
    os.makedirs(_d, exist_ok=True)

# Paylaşım linki görüntüleyici (auth bypass, salt okunur)
_share_qp = st.query_params.get("share")
if ENABLE_SHARE_LINKS and _share_qp:
    _share_link = resolve_share_token(str(_share_qp))
    if _share_link is None:
        st.error(t("share_invalid"))
        st.stop()
    _shared_session = load_shared_session(_share_link)
    if _shared_session is None:
        st.error(t("share_invalid"))
        st.stop()
    st.subheader(t("share_view_title"))
    st.caption(
        f"{_shared_session.get('title') or 'Sohbet'} · id={_shared_session.get('id')}"
    )
    for _m in _shared_session.get("messages") or []:
        with st.chat_message(_m.get("role") or "assistant"):
            st.markdown(_m.get("content") or "")
            if _m.get("sources"):
                with st.expander(t("sources")):
                    for _src in _m["sources"]:
                        _label = _src.get("label") or _src.get("source_file") or "?"
                        _score = _src.get("score")
                        if _score is not None:
                            st.markdown(f"- {_label} (skor: {float(_score):.3f})")
                        else:
                            st.markdown(f"- {_label}")
    st.stop()

# --- Auth (opsiyonel; indeks paylaşımlı, sohbetler kullanıcıya özel) ---
current_user = None
if auth_enabled():
    ensure_users_file()
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    # OIDC callback: ?code=&state=
    if oidc_enabled() and st.session_state["auth_user"] is None:
        qp = st.query_params
        code = qp.get("code")
        state = qp.get("state")
        if code and state:
            try:
                cfg = load_oidc_config()
                expected_state = st.session_state.get("oidc_state") or ""
                expected_nonce = st.session_state.get("oidc_nonce")
                code_verifier = st.session_state.get("oidc_code_verifier")
                result = complete_login(
                    cfg,
                    code=str(code),
                    expected_state=str(expected_state),
                    received_state=str(state),
                    expected_nonce=expected_nonce,
                    code_verifier=code_verifier,
                )
                user = upsert_oidc_user(
                    result["username"],
                    role=result["role"],
                    tenant_id=result["tenant_id"],
                    auto_provision=cfg.auto_provision,
                    claims=result.get("claims"),
                )
                write_audit(
                    "login_success",
                    username=user.username,
                    tenant_id=user.tenant_id if ENABLE_TENANTS else None,
                    details={"provider": "oidc", "role": user.role},
                )
                st.session_state["auth_user"] = user
                for k in (
                    "chat_session",
                    "chat_session_id",
                    "messages",
                    "bm25",
                    "bm25_index_id",
                    "oidc_state",
                    "oidc_nonce",
                    "oidc_code_verifier",
                ):
                    st.session_state.pop(k, None)
                # callback query parametrelerini temizle
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                st.rerun()
            except (OIDCError, ValueError) as exc:
                write_audit(
                    "login_fail",
                    username=None,
                    details={"reason": "oidc_error", "error": str(exc)},
                )
                st.error(f"SSO girişi başarısız: {exc}")
                try:
                    st.query_params.clear()
                except Exception:
                    pass

    if st.session_state["auth_user"] is None:
        st.subheader(t("login"))
        st.caption(
            "Çok kullanıcılı mod açık. Sohbet geçmişi kullanıcıya özeldir; "
            "indeks paylaşımlı veya kişisel olabilir."
            if get_language() == "tr"
            else "Multi-user mode is on. Chat history is per user; the index may be shared or private."
        )
        if ENABLE_TENANTS:
            st.caption(f"Tenant izolasyonu açık (varsayılan tenant: `{DEFAULT_TENANT}`).")

        if oidc_enabled():
            cfg = load_oidc_config()
            if not cfg.configured:
                st.warning(
                    "OIDC açık ancak `RAG_OIDC_ISSUER` / `RAG_OIDC_CLIENT_ID` / "
                    "`RAG_OIDC_REDIRECT_URI` eksik."
                )
            else:
                st.markdown("**Kurumsal giriş (SSO / OIDC)**" if get_language() == "tr" else "**Enterprise login (SSO / OIDC)**")
                if st.button(t("sso_login"), type="primary", key="oidc_login_btn"):
                    try:
                        disc = fetch_discovery(cfg.issuer)
                        state = secrets.token_urlsafe(24)
                        nonce = secrets.token_urlsafe(24)
                        verifier, challenge = generate_pkce_pair()
                        st.session_state["oidc_state"] = state
                        st.session_state["oidc_nonce"] = nonce
                        st.session_state["oidc_code_verifier"] = verifier
                        url = build_authorize_url(
                            disc, cfg, state=state, nonce=nonce, code_challenge=challenge
                        )
                        st.markdown(
                            f'<meta http-equiv="refresh" content="0;url={url}">',
                            unsafe_allow_html=True,
                        )
                        st.link_button("IdP'ye git (yönlendirilmezseniz)", url)
                        st.stop()
                    except Exception as exc:
                        st.error(f"OIDC başlatılamadı: {exc}")

        if not OIDC_ONLY:
            with st.form("login_form"):
                lu = st.text_input(t("username"))
                lp = st.text_input(t("password"), type="password")
                submitted = st.form_submit_button(t("login_btn"))
            if submitted:
                user = authenticate(lu, lp)
                if user is None:
                    write_audit(
                        "login_fail", username=lu, details={"reason": "invalid_credentials"}
                    )
                    st.error(t("invalid_credentials"))
                else:
                    write_audit(
                        "login_success",
                        username=user.username,
                        tenant_id=user.tenant_id if ENABLE_TENANTS else None,
                        details={"provider": "local"},
                    )
                    st.session_state["auth_user"] = user
                    for k in (
                        "chat_session",
                        "chat_session_id",
                        "messages",
                        "bm25",
                        "bm25_index_id",
                    ):
                        st.session_state.pop(k, None)
                    st.rerun()
            st.info(
                "Örnek: `users.example.json` dosyasını `metadata/users.json` olarak "
                "kopyalayın veya `RAG_AUTH_BOOTSTRAP_ADMIN` kullanın."
            )
        elif not oidc_enabled():
            st.error("`RAG_OIDC_ONLY=1` ama OIDC kapalı. `RAG_ENABLE_OIDC=1` ayarlayın.")
        st.stop()

    current_user = st.session_state["auth_user"]

ws = resolve_workspace(current_user)
ensure_workspace_dirs(ws)
active_chat_dir = ws.chat_dir
active_data_dir = ws.data_dir
active_index_path = ws.index_path
active_docstore_path = ws.docstore_path
active_meta_path = ws.source_meta_path
can_ingest = (not auth_enabled()) or bool(current_user and current_user.can_ingest)

# Kullanıcı/workspace değişince indeks oturumunu yenile
if st.session_state.get("workspace_key") != ws.key:
    st.session_state["workspace_key"] = ws.key
    st.session_state.pop("index", None)
    st.session_state.pop("bm25", None)
    st.session_state.pop("bm25_index_id", None)
    for k in ("chat_session", "chat_session_id", "messages"):
        # chat zaten kullanıcıya özel; yine de workspace değişiminde temizle
        if current_user is not None:
            st.session_state.pop(k, None)


def _source_anchor(source_file: str, chunk_id: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_file).strip("-").lower()
    return f"src-{safe}-c{chunk_id}"


@st.cache_resource(show_spinner=True)
def get_embedder(model_name: str):
    return Embedder(model_name=model_name)


@st.cache_resource(show_spinner=True)
def get_cached_reranker(enabled: bool, model_name: str):
    if not enabled:
        return None
    return get_reranker(model_name=model_name, prefer_cross_encoder=True)


@st.cache_resource(show_spinner=False)
def load_or_create_index(
    dim: int,
    embedding_model: str,
    index_path: str,
    docstore_path: str,
    workspace_key: str,
    backend: str,
):
    # workspace_key + backend cache ayrımı için
    _ = (workspace_key, backend)
    try:
        if os.path.exists(docstore_path) or (
            backend == "faiss" and os.path.exists(index_path) and os.path.exists(docstore_path)
        ):
            loaded = load_index(
                index_path,
                docstore_path,
                backend=backend,
                dim=dim,
                embedding_model=embedding_model,
            )
            if loaded.dim == dim and (
                not loaded.embedding_model or loaded.embedding_model == embedding_model
            ):
                if not loaded.embedding_model:
                    loaded.embedding_model = embedding_model
                if loaded.size > 0 or os.path.exists(docstore_path):
                    return loaded
    except Exception:
        pass
    return create_index(dim=dim, embedding_model=embedding_model, backend=backend)


# Sidebar
with st.sidebar:
    st.header(t("settings"))
    lang_choice = st.selectbox(
        t("language"),
        options=list(SUPPORTED_LANGS),
        index=list(SUPPORTED_LANGS).index(get_language())
        if get_language() in SUPPORTED_LANGS
        else 0,
        format_func=lambda c: "Türkçe" if c == "tr" else "English",
        key="ui_lang_select",
    )
    if lang_choice != st.session_state.get("ui_lang"):
        st.session_state["ui_lang"] = lang_choice
        set_language(lang_choice)
        st.rerun()
    set_language(st.session_state["ui_lang"])
    st.caption(f"{t('vector_backend')}: `{vector_backend()}`")
    if ENABLE_PROMETHEUS:
        st.caption(f"Prometheus scrape: `:{PROMETHEUS_PORT}/metrics`")
    if ENABLE_COLLAB_NOTES:
        _nc_user = current_user.username if current_user else "local"
        with st.expander(t("collab_notify_center"), expanded=False):
            _global_notifs = list_notifications_global(_nc_user, unread_only=True, limit=30)
            if _global_notifs:
                for _gn in reversed(_global_notifs):
                    st.caption(
                        t(
                            "collab_mention_notify_ws",
                            workspace=_gn.get("workspace_key") or "-",
                            from_user=_gn.get("from_user") or "-",
                            preview=_gn.get("body_preview") or "",
                        )
                    )
                if st.button(
                    t("collab_mark_read_all"),
                    key="collab_mark_read_all",
                    use_container_width=True,
                ):
                    mark_notifications_read_global(_nc_user)
                    st.rerun()
                if st.button(
                    t("collab_send_digest"),
                    key="collab_send_digest",
                    use_container_width=True,
                ):
                    from rag.collab_notify_digest import send_digest_email

                    _dig = send_digest_email(_nc_user)
                    if _dig.get("sent"):
                        st.success(t("collab_digest_sent", count=_dig.get("count") or 0))
                    else:
                        st.info(t("collab_digest_skip", reason=_dig.get("reason") or "?"))
            else:
                st.caption(t("collab_no_notifications"))
            from rag.collab_notify_digest import (
                export_digest_report_csv,
                list_digest_report_tenants,
                list_digest_report_users,
                summarize_digest_report,
            )

            with st.expander(t("collab_digest_report"), expanded=False):
                _tenant_opts = ["(all)"] + list_digest_report_tenants()
                try:
                    from rag.auth import list_users_detail

                    for _u in list_users_detail():
                        _tid = str((_u or {}).get("tenant_id") or "").strip()
                        if _tid and _tid not in _tenant_opts:
                            _tenant_opts.append(_tid)
                except Exception:
                    pass
                _sel_tenant = st.selectbox(
                    t("collab_digest_report_tenant"),
                    _tenant_opts,
                    key="digest_report_tenant",
                )
                _filter_tid = None if _sel_tenant == "(all)" else _sel_tenant
                _user_opts = ["(all)"] + list_digest_report_users(tenant_id=_filter_tid)
                _sel_user = st.selectbox(
                    t("collab_digest_report_user"),
                    _user_opts,
                    key="digest_report_user",
                )
                _filter_user = None if _sel_user == "(all)" else _sel_user
                _dcols = st.columns(2)
                with _dcols[0]:
                    _since = st.text_input(
                        t("collab_digest_report_since"),
                        key="digest_report_since",
                        placeholder="2026-01-01T00:00:00+00:00",
                    )
                with _dcols[1]:
                    _until = st.text_input(
                        t("collab_digest_report_until"),
                        key="digest_report_until",
                        placeholder="2026-12-31T23:59:59+00:00",
                    )
                _since_v = _since.strip() or None
                _until_v = _until.strip() or None
                _rep = summarize_digest_report(
                    limit=100,
                    tenant_id=_filter_tid,
                    username=_filter_user,
                    since=_since_v,
                    until=_until_v,
                )
                st.caption(
                    t(
                        "collab_digest_report_summary",
                        total=_rep.get("total") or 0,
                        sent=_rep.get("sent") or 0,
                        skipped=_rep.get("skipped") or 0,
                    )
                )
                _ch = _rep.get("channels") or {}
                st.caption(
                    t(
                        "collab_digest_report_channels",
                        email=_ch.get("email") or 0,
                        webhook=_ch.get("webhook") or 0,
                        push=_ch.get("push") or 0,
                    )
                )
                _reasons = _rep.get("by_reason") or {}
                if _reasons:
                    st.json(_reasons)
                for _row in (_rep.get("recent") or [])[-5:]:
                    st.caption(
                        f"{_row.get('ts') or '-'} · {_row.get('username') or '-'} · "
                        f"tenant={_row.get('tenant_id') or '-'} · "
                        f"sent={_row.get('sent')} · count={_row.get('count')} · "
                        f"reason={_row.get('reason') or '-'}"
                    )
                _csv = export_digest_report_csv(
                    limit=500,
                    tenant_id=_filter_tid,
                    username=_filter_user,
                    since=_since_v,
                    until=_until_v,
                )
                st.download_button(
                    t("collab_digest_export_csv"),
                    data=_csv,
                    file_name="digest_report.csv",
                    mime="text/csv",
                    key="digest_report_csv_dl",
                    use_container_width=True,
                )
                if st.button(
                    t("collab_digest_alert_check"),
                    key="digest_alert_check_btn",
                    use_container_width=True,
                ):
                    from rag.collab_notify_digest import check_digest_alerts

                    _alert = check_digest_alerts(
                        tenant_id=_filter_tid,
                        username=_filter_user,
                        since=_since_v,
                        until=_until_v,
                        limit=500,
                        dry_run=True,
                    )
                    if _alert.get("fired"):
                        st.warning(
                            t(
                                "collab_digest_alert_fired",
                                reasons=", ".join((_alert.get("alert") or {}).get("reasons") or []),
                            )
                        )
                        st.json(_alert.get("alert") or {})
                    else:
                        st.success(t("collab_digest_alert_ok"))
            from rag.collab_notify_push import (
                generate_vapid_keys,
                load_service_worker_js,
                parse_webpush_subscription,
                prune_expired_tokens,
                register_device_token,
                revoke_device_token,
                summarize_user_devices,
                vapid_configured,
                vapid_public_key,
            )

            st.markdown(t("collab_push_devices"))
            if current_user and current_user.role == "admin":
                with st.expander(t("collab_push_vapid_generate"), expanded=False):
                    if st.button(t("collab_push_vapid_generate_btn"), key="vapid_gen"):
                        _vk = generate_vapid_keys()
                        st.session_state["vapid_generated"] = _vk
                    _vk_show = st.session_state.get("vapid_generated")
                    if _vk_show:
                        st.caption(t("collab_push_vapid_generate_hint"))
                        st.code(_vk_show.get("env") or "")
            if vapid_configured():
                with st.expander(t("collab_push_browser"), expanded=False):
                    st.caption(t("collab_push_browser_help"))
                    try:
                        from rag.collab_http import collab_http_public_base

                        _http_base = collab_http_public_base()
                        st.markdown(
                            t(
                                "collab_push_pwa_link",
                                url=f"{_http_base}/webpush?user={_nc_user}",
                            )
                        )
                    except Exception:
                        _http_base = ""
                    st.code(vapid_public_key()[:64] + ("…" if len(vapid_public_key()) > 64 else ""))
                    _sw = load_service_worker_js()
                    if _sw:
                        st.download_button(
                            t("collab_push_sw_download"),
                            data=_sw,
                            file_name="sw.js",
                            mime="application/javascript",
                            key="webpush_sw_dl",
                            use_container_width=True,
                        )
                    try:
                        from rag.webpush_browser import render_webpush_subscribe_widget

                        render_webpush_subscribe_widget(username=_nc_user)
                    except Exception as _wp_exc:
                        st.caption(f"{t('collab_push_browser_error')}: {_wp_exc}")
                    _bridge = st.text_area(
                        t("collab_push_bridge_json"),
                        key="webpush_bridge_json",
                        height=100,
                        help=t("collab_push_bridge_help"),
                    )
                    if st.button(
                        t("collab_push_bridge_save"),
                        key="webpush_bridge_save",
                        use_container_width=True,
                    ):
                        if parse_webpush_subscription(_bridge or ""):
                            register_device_token(
                                _nc_user,
                                _bridge.strip(),
                                platform="webpush",
                                label="streamlit-bridge",
                                device_name="Browser",
                                os_name="web",
                            )
                            st.success(t("collab_push_bridge_ok"))
                            st.rerun()
                        else:
                            st.warning(t("collab_push_bridge_invalid"))
            else:
                st.caption(t("collab_push_vapid_missing"))
            _devices = summarize_user_devices(_nc_user)
            if _devices:
                for _dv in _devices:
                    _status = (
                        t("collab_push_expired")
                        if _dv.get("expired") or _dv.get("revoked")
                        else t("collab_push_active")
                    )
                    _os = " ".join(
                        x for x in [_dv.get("os_name"), _dv.get("os_version")] if x
                    ) or "-"
                    st.caption(
                        f"{_dv.get('device_name') or _dv.get('label') or '-'} · "
                        f"{_dv.get('platform')} · {_dv.get('token_preview')} · "
                        f"{_os} · qh={_dv.get('quiet_hours') or '-'} · "
                        f"last={_dv.get('last_seen_at') or '-'} · {_status}"
                    )
                    if not _dv.get("revoked") and st.button(
                        t("collab_push_revoke"),
                        key=f"push_rev_{_dv.get('token_preview')}",
                        use_container_width=True,
                    ):
                        revoke_device_token(_nc_user, str(_dv.get("token") or ""))
                        st.rerun()
            else:
                st.caption(t("collab_push_no_devices"))
            _new_tok = st.text_input(t("collab_push_token"), key="push_token_input")
            _new_plat = st.selectbox(
                t("collab_push_platform"),
                ["fcm", "apns", "generic", "webpush"],
                key="push_platform_input",
            )
            _new_label = st.text_input(t("collab_push_label"), key="push_label_input")
            _new_dname = st.text_input(t("collab_push_device_name"), key="push_dname_input")
            _new_os = st.text_input(t("collab_push_os_name"), key="push_os_input")
            _new_osv = st.text_input(t("collab_push_os_version"), key="push_osv_input")
            _new_appv = st.text_input(t("collab_push_app_version"), key="push_appv_input")
            _new_qh = st.text_input(
                t("collab_push_quiet_hours"),
                key="push_qh_input",
                placeholder="22:00-07:00 | off",
                help=t("collab_push_quiet_hours_help"),
            )
            _geo_cols = st.columns(3)
            with _geo_cols[0]:
                _geo_lat = st.text_input(t("collab_push_geofence_lat"), key="push_geo_lat")
            with _geo_cols[1]:
                _geo_lon = st.text_input(t("collab_push_geofence_lon"), key="push_geo_lon")
            with _geo_cols[2]:
                _geo_r = st.text_input(
                    t("collab_push_geofence_radius"),
                    key="push_geo_r",
                    placeholder="500",
                )
            cpush1, cpush2 = st.columns(2)
            with cpush1:
                if st.button(t("collab_push_register"), use_container_width=True, key="push_reg"):
                    if _new_tok.strip():
                        _kw = dict(
                            platform=_new_plat,
                            label=_new_label or None,
                            device_name=_new_dname or None,
                            os_name=_new_os or None,
                            os_version=_new_osv or None,
                            app_version=_new_appv or None,
                            quiet_hours=_new_qh.strip() if _new_qh.strip() else None,
                        )
                        try:
                            if _geo_lat.strip() and _geo_lon.strip():
                                _kw["geofence_lat"] = float(_geo_lat)
                                _kw["geofence_lon"] = float(_geo_lon)
                                if _geo_r.strip():
                                    _kw["geofence_radius_m"] = float(_geo_r)
                                _kw["last_lat"] = float(_geo_lat)
                                _kw["last_lon"] = float(_geo_lon)
                        except ValueError:
                            st.warning(t("collab_push_geofence_invalid"))
                        else:
                            register_device_token(_nc_user, _new_tok.strip(), **_kw)
                            st.rerun()
                    else:
                        st.warning(t("collab_push_token_required"))
            with cpush2:
                if st.button(t("collab_push_prune"), use_container_width=True, key="push_prune"):
                    prune_expired_tokens(username=_nc_user)
                    st.rerun()
    provider_options = ["ollama", "openai"]
    provider_index = (
        provider_options.index(DEFAULT_LLM_PROVIDER)
        if DEFAULT_LLM_PROVIDER in provider_options
        else 0
    )
    provider = st.selectbox("LLM Sağlayıcı", options=provider_options, index=provider_index)
    if provider == "ollama":
        model_name = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)
        st.caption("Öneri: küçük/quantized model (örn. phi3:mini) CPU'da daha hızlı")
        try:
            r = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=1.5)
            if r.status_code == 200:
                st.success(f"Ollama çalışıyor ({OLLAMA_HOST})")
            else:
                st.warning("Ollama'a ulaşılamadı veya beklenmeyen yanıt.")
        except Exception:
            st.warning("Ollama kapalı görünüyor. Lütfen Ollama'yı başlatın.")
    else:
        model_name = st.text_input("OpenAI Model", value=DEFAULT_OPENAI_MODEL)
        st.caption("OPENAI_API_KEY çevre değişkeni gerekli")
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY tanımlı değil. Ayarlamazsanız yanıt üretemeyiz.")

    embedding_keys = list(EMBEDDING_PRESETS.keys())
    default_preset = default_embedding_preset()
    if default_preset not in embedding_keys:
        default_preset = preset_for_model(DEFAULT_EMBEDDING_MODEL)
    embedding_preset = st.selectbox(
        "Embedding modeli",
        options=embedding_keys,
        index=embedding_keys.index(default_preset) if default_preset in embedding_keys else 0,
        format_func=lambda k: EMBEDDING_PRESETS[k]["label"],
    )
    embedding_model = resolve_embedding_model(embedding_preset)
    st.caption(embedding_model)

    top_k = st.slider("Top‑K", min_value=3, max_value=10, value=DEFAULT_TOP_K)
    use_hybrid = st.checkbox(t("hybrid_search"), value=True)
    hybrid_alpha = st.slider(
        "Hybrid α (vektör ağırlığı)",
        min_value=0.0,
        max_value=1.0,
        value=float(HYBRID_ALPHA),
        step=0.05,
        disabled=not use_hybrid,
    )
    use_reranker = st.checkbox(t("reranker"), value=ENABLE_RERANKER)
    if use_reranker:
        st.caption(f"Model: {DEFAULT_RERANKER_MODEL}")
    use_stream = st.checkbox(t("stream_answer"), value=True)
    rewrite_options = {
        "none": t("rewrite_none"),
        "hyde": t("rewrite_hyde"),
        "expand": t("rewrite_expand"),
        "hyde+expand": t("rewrite_both"),
    }
    default_rw = DEFAULT_QUERY_REWRITE_MODE if ENABLE_QUERY_REWRITE else "none"
    if default_rw not in rewrite_options:
        default_rw = "hyde" if ENABLE_QUERY_REWRITE else "none"
    rewrite_mode = st.selectbox(
        t("query_rewrite"),
        options=list(rewrite_options.keys()),
        index=list(rewrite_options.keys()).index(default_rw),
        format_func=lambda k: rewrite_options[k],
    )
    use_agent_tools = st.checkbox(t("agent_tools"), value=ENABLE_AGENT_TOOLS)
    use_source_highlight = st.checkbox(t("source_highlight"), value=ENABLE_SOURCE_HIGHLIGHT)
    use_agent_memory = st.checkbox(t("agent_memory"), value=ENABLE_AGENT_MEMORY)
    use_agent_planner = st.checkbox(t("agent_planner"), value=ENABLE_AGENT_PLANNER)
    use_vision_llm = st.checkbox(t("vision_llm"), value=ENABLE_VISION_LLM)
    use_profile_memory = st.checkbox(t("profile_memory"), value=ENABLE_PROFILE_MEMORY)
    rebuild = False
    if can_ingest:
        rebuild = st.button(t("rebuild_index"))
    else:
        st.caption("İndeks yönetimi için admin yetkisi gerekir.")

    if current_user:
        st.header(t("account"))
        st.write(f"Kullanıcı: **{current_user.username}** (`{current_user.role}`)")
        if ENABLE_TENANTS:
            st.caption(f"Tenant: `{current_user.tenant_id}`")
        from rag.auth import get_user_timezone, update_user_timezone
        from rag.auth import get_user_quiet_hours, update_user_quiet_hours

        _cur_tz = get_user_timezone(current_user.username) or ""
        _tz_opts = [
            "",
            "UTC",
            "Europe/Istanbul",
            "Europe/London",
            "Europe/Berlin",
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
        ]
        _tz_index = _tz_opts.index(_cur_tz) if _cur_tz in _tz_opts else 0
        _new_tz = st.selectbox(
            t("account_timezone"),
            _tz_opts,
            index=_tz_index,
            format_func=lambda x: t("account_timezone_default") if x == "" else x,
            key="account_timezone_select",
        )
        if st.button(t("account_timezone_save"), use_container_width=True, key="save_tz"):
            update_user_timezone(current_user.username, _new_tz or None)
            st.success(t("account_timezone_saved", tz=_new_tz or "default"))
        _cur_qh = get_user_quiet_hours(current_user.username) or ""
        _new_qh = st.text_input(
            t("account_quiet_hours"),
            value=_cur_qh,
            placeholder="22:00-07:00",
            key="account_quiet_hours_input",
            help=t("account_quiet_hours_help"),
        )
        if st.button(t("account_quiet_hours_save"), use_container_width=True, key="save_qh"):
            try:
                update_user_quiet_hours(
                    current_user.username,
                    _new_qh.strip() or None,
                )
                st.success(
                    t(
                        "account_quiet_hours_saved",
                        qh=_new_qh.strip() or t("account_quiet_hours_default"),
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
        from rag.auth import get_user_digest_channels, update_user_digest_channels

        _ch = get_user_digest_channels(current_user.username)
        st.caption(t("account_digest_channels"))
        _ch_email = st.checkbox(
            t("account_digest_email"),
            value=bool(_ch.get("email", True)),
            key="account_digest_email",
        )
        _ch_webhook = st.checkbox(
            t("account_digest_webhook"),
            value=bool(_ch.get("webhook", True)),
            key="account_digest_webhook",
        )
        _ch_push = st.checkbox(
            t("account_digest_push"),
            value=bool(_ch.get("push", True)),
            key="account_digest_push",
        )
        if st.button(t("account_digest_channels_save"), use_container_width=True, key="save_ch"):
            update_user_digest_channels(
                current_user.username,
                {
                    "email": _ch_email,
                    "webhook": _ch_webhook,
                    "push": _ch_push,
                },
            )
            st.success(t("account_digest_channels_saved"))
        from rag.auth import get_user_quiet_channels, update_user_quiet_channels

        _qc = get_user_quiet_channels(current_user.username)
        st.caption(t("account_quiet_channels"))
        st.caption(t("account_quiet_channels_help"))
        _qc_email = st.checkbox(
            t("account_quiet_ch_email"),
            value=bool(_qc.get("email", True)),
            key="account_quiet_ch_email",
        )
        _qc_webhook = st.checkbox(
            t("account_quiet_ch_webhook"),
            value=bool(_qc.get("webhook", True)),
            key="account_quiet_ch_webhook",
        )
        _qc_push = st.checkbox(
            t("account_quiet_ch_push"),
            value=bool(_qc.get("push", False)),
            key="account_quiet_ch_push",
        )
        if st.button(t("account_quiet_channels_save"), use_container_width=True, key="save_qc"):
            update_user_quiet_channels(
                current_user.username,
                {
                    "email": _qc_email,
                    "webhook": _qc_webhook,
                    "push": _qc_push,
                },
            )
            st.success(t("account_quiet_channels_saved"))
        if ws.shared:
            st.caption("İndeks paylaşımlı (aynı tenant içindeki kullanıcılar).")
        else:
            st.caption(f"Kişisel indeks: `{ws.data_dir}`")
        if st.button(t("logout")):
            write_audit(
                "logout",
                username=current_user.username,
                tenant_id=ws.tenant_id,
                path=ws.audit_path,
            )
            st.session_state["auth_user"] = None
            for k in ("chat_session", "chat_session_id", "messages", "index", "bm25", "bm25_index_id", "workspace_key"):
                st.session_state.pop(k, None)
            st.rerun()

        if current_user.role == "admin":
            st.header(t("admin"))
            users = list_users_detail()
            st.caption(f"{len(users)} kullanıcı")
            for u in users:
                acl_f = u.get("allowed_folders")
                acl_t = u.get("allowed_tags")
                acl_txt = f" tenant={u.get('tenant_id') or DEFAULT_TENANT}"
                if u.get("auth_provider"):
                    acl_txt += f" auth={u.get('auth_provider')}"
                if acl_f is not None:
                    acl_txt += f" klasör={acl_f}"
                if acl_t is not None:
                    acl_txt += f" etiket={acl_t}"
                st.write(f"- `{u['username']}` ({u['role']}){acl_txt}")

            with st.expander("Kullanıcı ekle", expanded=False):
                nu = st.text_input("Yeni kullanıcı adı", key="admin_new_user")
                npw = st.text_input("Parola", type="password", key="admin_new_pass")
                nrole = st.selectbox("Rol", options=["user", "admin"], key="admin_new_role")
                ntenant = st.text_input("Tenant", value=DEFAULT_TENANT, key="admin_new_tenant")
                nfolders = st.text_input(
                    "İzinli klasörler (* = tümü / boş = tümü)",
                    key="admin_new_folders",
                    placeholder="hukuk,genel",
                )
                ntags = st.text_input(
                    "İzinli etiketler (* = tümü / boş = tümü)",
                    key="admin_new_tags",
                    placeholder="public",
                )
                if st.button("Ekle", key="admin_add_btn"):
                    try:
                        folders = None if not nfolders.strip() or nfolders.strip() == "*" else nfolders
                        tags = None if not ntags.strip() or ntags.strip() == "*" else ntags
                        created = add_user(
                            nu,
                            npw,
                            role=nrole,
                            tenant_id=ntenant,
                            allowed_folders=folders,
                            allowed_tags=tags,
                        )
                        write_audit(
                            "admin_add_user",
                            username=current_user.username,
                            tenant_id=ws.tenant_id,
                            details={"created": created.username, "role": created.role, "tenant": created.tenant_id},
                            path=ws.audit_path,
                        )
                        st.success(f"Eklendi: {created.username} ({created.role})")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with st.expander("ACL güncelle", expanded=False):
                acl_user = st.selectbox(
                    "Kullanıcı",
                    options=[u["username"] for u in users],
                    key="admin_acl_user",
                )
                acl_folders = st.text_input(
                    "Klasörler (* = tümü)",
                    key="admin_acl_folders",
                    placeholder="hukuk,genel",
                )
                acl_tags = st.text_input(
                    "Etiketler (* = tümü)",
                    key="admin_acl_tags",
                    placeholder="public",
                )
                if st.button("ACL kaydet", key="admin_acl_save"):
                    try:
                        clear_f = acl_folders.strip() in {"", "*"}
                        clear_t = acl_tags.strip() in {"", "*"}
                        updated = update_user_acl(
                            acl_user,
                            allowed_folders=None if clear_f else acl_folders,
                            allowed_tags=None if clear_t else acl_tags,
                            clear_folders=clear_f,
                            clear_tags=clear_t,
                        )
                        write_audit(
                            "admin_acl_update",
                            username=current_user.username,
                            tenant_id=ws.tenant_id,
                            details={
                                "target": updated.username,
                                "folders": updated.allowed_folders,
                                "tags": updated.allowed_tags,
                            },
                            path=ws.audit_path,
                        )
                        st.success(
                            f"ACL güncellendi: {updated.username} "
                            f"folders={updated.allowed_folders} tags={updated.allowed_tags}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with st.expander("Kullanıcı sil", expanded=False):
                del_opts = [u["username"] for u in users if u["username"] != current_user.username]
                if not del_opts:
                    st.caption("Silinebilir başka kullanıcı yok.")
                else:
                    victim = st.selectbox("Silinecek", options=del_opts, key="admin_del_user")
                    if st.button("Sil", key="admin_del_btn"):
                        try:
                            deleted = delete_user(victim)
                            write_audit(
                                "admin_delete_user",
                                username=current_user.username,
                                tenant_id=ws.tenant_id,
                                details={"deleted": deleted},
                                path=ws.audit_path,
                            )
                            st.success(f"Silindi: {deleted}")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

            with st.expander("Audit log (son 20)", expanded=False):
                rows = read_audit(path=ws.audit_path, limit=20)
                if not rows:
                    st.caption("Kayıt yok.")
                else:
                    for row in rows:
                        st.write(
                            f"`{row.get('ts')}` · **{row.get('event')}** · "
                            f"{row.get('username') or '-'} · {row.get('details')}"
                        )

            with st.expander("Metrikler (observability)", expanded=False):
                if not ENABLE_METRICS:
                    st.caption("Metrikler kapalı (`RAG_ENABLE_METRICS=0`).")
                else:
                    summary = summarize_metrics(path=ws.metrics_path)
                    q = summary["query"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sorgu", q["count"])
                    c2.metric("No-answer oranı", f"{q['no_answer_rate']:.0%}")
                    c3.metric("Ort. gate skoru", f"{q['avg_gate_score']:.3f}")
                    c4.metric("Ort. gecikme (ms)", f"{q['avg_latency_ms']:.0f}")
                    st.caption(
                        f"Toplam olay: {summary['total_events']} · "
                        f"Ingest: {summary['ingest']['count']} "
                        f"({summary['ingest']['chunks_added']} chunk) · "
                        f"Rebuild: {summary['rebuild']['count']} · "
                        f"Silme: {summary['delete']['count']}"
                    )
                    jg = summary.get("judge") or {}
                    if jg.get("runs"):
                        st.caption(
                            t(
                                "metrics_judge_summary",
                                runs=jg.get("runs") or 0,
                                ok=jg.get("ok") or 0,
                                failed=jg.get("failed") or 0,
                                soft_fail=jg.get("soft_fail") or 0,
                                avg=f"{float(jg.get('avg_accuracy') or 0):.2%}",
                            )
                        )
                        st.caption(
                            t(
                                "metrics_judge_soft_fail_rate",
                                rate=f"{float(jg.get('soft_fail_rate') or 0):.0%}",
                            )
                        )
                        _accs = jg.get("recent_accuracies") or []
                        if len(_accs) >= 2:
                            st.caption(t("metrics_judge_trend"))
                            st.line_chart({"accuracy": _accs})
                    if summary["recent"]:
                        st.markdown("**Son kayıtlar**")
                        for row in reversed(summary["recent"]):
                            vals = row.get("values") or {}
                            st.write(
                                f"`{row.get('ts')}` · **{row.get('kind')}** · "
                                f"{row.get('username') or '-'} · {vals}"
                            )

    st.header(t("chats"))
    sessions = list_sessions(chat_dir=active_chat_dir)
    session_ids = [s["id"] for s in sessions]
    labels = {
        s["id"]: f"{s['title']} ({s['n_messages']})"
        for s in sessions
    }

    if "chat_session_id" not in st.session_state:
        if session_ids:
            st.session_state["chat_session_id"] = session_ids[0]
        else:
            created = create_session(chat_dir=active_chat_dir)
            st.session_state["chat_session_id"] = created["id"]
            sessions = list_sessions(chat_dir=active_chat_dir)
            session_ids = [s["id"] for s in sessions]
            labels = {s["id"]: f"{s['title']} ({s['n_messages']})" for s in sessions}

    current_id = st.session_state["chat_session_id"]
    if current_id not in session_ids and session_ids:
        current_id = session_ids[0]
        st.session_state["chat_session_id"] = current_id

    selected = st.selectbox(
        "Kayıtlı oturum",
        options=session_ids,
        index=session_ids.index(current_id) if current_id in session_ids else 0,
        format_func=lambda sid: labels.get(sid, sid),
    )
    if selected != st.session_state.get("chat_session_id"):
        st.session_state["chat_session_id"] = selected
        loaded = load_session(selected, chat_dir=active_chat_dir) or create_session(
            chat_dir=active_chat_dir
        )
        st.session_state["chat_session"] = loaded
        st.session_state["messages"] = list(loaded.get("messages") or [])
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        new_chat = st.button(t("new_chat"), use_container_width=True)
    with c2:
        clear_chat = st.button(t("clear_chat"), use_container_width=True)
    with c3:
        delete_chat = st.button(t("delete"), use_container_width=True)

    if ENABLE_SHARE_LINKS:
        st.caption(t("share_link_caption"))
        _sid = st.session_state.get("chat_session_id")
        _active_links = (
            list_links_for_session(active_chat_dir, _sid) if _sid else []
        )
        if st.button(t("share_link"), use_container_width=True, key="create_share_link"):
            if _sid:
                _new_link = create_share_link(
                    active_chat_dir,
                    _sid,
                    owner=current_user.username if current_user else None,
                    ttl_days=SHARE_LINK_DEFAULT_TTL_DAYS,
                )
                st.session_state["last_share_url"] = build_share_url(
                    _new_link.token, PUBLIC_BASE_URL
                )
                write_audit(
                    "share_link_create",
                    username=current_user.username if current_user else None,
                    tenant_id=ws.tenant_id if ENABLE_TENANTS else None,
                    details={"session_id": _sid, "token": _new_link.token[:8]},
                )
        if st.session_state.get("last_share_url"):
            st.code(st.session_state["last_share_url"])
        if _active_links:
            if st.button(
                t("share_link_revoke"),
                use_container_width=True,
                key="revoke_share_link",
            ):
                revoke_share_link(_active_links[0].token)
                st.session_state.pop("last_share_url", None)
                write_audit(
                    "share_link_revoke",
                    username=current_user.username if current_user else None,
                    tenant_id=ws.tenant_id if ENABLE_TENANTS else None,
                    details={"session_id": _sid, "token": _active_links[0].token[:8]},
                )
                st.rerun()

    if ENABLE_COLLAB_NOTES:
        with st.expander(t("collab_note"), expanded=False):
            if ENABLE_COLLAB_CRDT:
                _crdt = load_crdt(ws.key)
                _note_content = _crdt.materialize()
                _note_revision = _crdt.revision
                _note_user = _crdt.updated_by
                _note_ts = _crdt.updated_at
                st.caption(t("collab_crdt_on"))
            else:
                _legacy = load_note(ws.key)
                _note_content = _legacy.content
                _note_revision = _legacy.revision
                _note_user = _legacy.updated_by
                _note_ts = _legacy.updated_at
            if f"collab_revision_{ws.key}" not in st.session_state:
                st.session_state[f"collab_revision_{ws.key}"] = _note_revision
            if _note_user or _note_ts:
                st.caption(
                    t(
                        "collab_updated",
                        user=_note_user or "-",
                        ts=_note_ts or "-",
                    )
                )
            if ENABLE_COLLAB_WS and websockets_available():
                _ws_url = f"ws://{COLLAB_WS_PUBLIC_HOST}:{COLLAB_WS_PORT}"
                st.caption(t("collab_ws_url", url=_ws_url))
                if ENABLE_COLLAB_LIVE_EDITOR:
                    st.caption(t("collab_live_editor"))
                    _uname = current_user.username if current_user else "local"
                    render_collab_live_editor(
                        ws_url=_ws_url,
                        workspace_key=ws.key,
                        username=_uname,
                        initial_content=_note_content,
                        revision=_note_revision,
                        richtext=ENABLE_COLLAB_RICHTEXT,
                    )
                else:
                    st.code(
                        json.dumps(
                            {"op": "join", "workspace_key": ws.key, "username": "alice"},
                            ensure_ascii=False,
                        ),
                        language="json",
                    )
            if ENABLE_COLLAB_CRDT:
                _polled = load_crdt(ws.key)
                _poll_rev = _polled.revision
                _poll_content = _polled.materialize()
            else:
                _polled = load_note(ws.key)
                _poll_rev = _polled.revision
                _poll_content = _polled.content
            if _poll_rev > st.session_state.get(f"collab_revision_{ws.key}", 0):
                st.info(t("collab_remote_update"))
            if st.button(
                t("collab_refresh"),
                use_container_width=True,
                key=f"collab_refresh_{ws.key}",
            ):
                st.session_state[f"collab_revision_{ws.key}"] = _poll_rev
                st.rerun()
            _collab_text = st.text_area(
                "collab",
                value=_note_content,
                height=120,
                label_visibility="collapsed",
                key=f"collab_text_{ws.key}",
            )
            if st.button(t("collab_save"), use_container_width=True, key=f"collab_save_{ws.key}"):
                uname = current_user.username if current_user else None
                if ENABLE_COLLAB_CRDT:
                    _saved = apply_text_edit_with_undo(ws.key, _collab_text, author=uname)
                    st.session_state[f"collab_revision_{ws.key}"] = _saved.revision
                    write_audit(
                        "collab_crdt_save",
                        username=uname,
                        tenant_id=ws.tenant_id if ENABLE_TENANTS else None,
                        details={"workspace": ws.key, "revision": _saved.revision},
                    )
                    st.success("Kaydedildi (CRDT birleştirme)")
                    st.rerun()
                else:
                    try:
                        _saved = save_note(
                            ws.key,
                            _collab_text,
                            username=uname,
                            expected_revision=st.session_state.get(f"collab_revision_{ws.key}"),
                        )
                        st.session_state[f"collab_revision_{ws.key}"] = _saved.revision
                        write_audit(
                            "collab_note_save",
                            username=uname,
                            tenant_id=ws.tenant_id if ENABLE_TENANTS else None,
                            details={"workspace": ws.key, "revision": _saved.revision},
                        )
                        st.success("Kaydedildi")
                    except ValueError:
                        st.warning(t("collab_conflict"))
            if ENABLE_COLLAB_CRDT:
                uc1, uc2 = st.columns(2)
                with uc1:
                    if st.button(t("collab_undo"), use_container_width=True, key=f"collab_undo_{ws.key}"):
                        _u = undo_edit(ws.key, author=current_user.username if current_user else None)
                        st.session_state[f"collab_revision_{ws.key}"] = _u.revision
                        st.rerun()
                with uc2:
                    if st.button(t("collab_redo"), use_container_width=True, key=f"collab_redo_{ws.key}"):
                        _r = redo_edit(ws.key, author=current_user.username if current_user else None)
                        st.session_state[f"collab_revision_{ws.key}"] = _r.revision
                        st.rerun()
            if ENABLE_COLLAB_CRDT and ENABLE_COLLAB_RICHTEXT:
                st.caption(t("collab_richtext"))
                _rt_html = render_rich_html(ws.key)
                if _rt_html:
                    st.markdown(_rt_html, unsafe_allow_html=True)
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    _mk_start = st.number_input(
                        t("collab_mark_start"),
                        min_value=0,
                        value=0,
                        key=f"mk_start_{ws.key}",
                    )
                with rc2:
                    _mk_end = st.number_input(
                        t("collab_mark_end"),
                        min_value=0,
                        value=min(5, max(0, len(_note_content))),
                        key=f"mk_end_{ws.key}",
                    )
                with rc3:
                    _mk_kind = st.selectbox(
                        t("collab_mark_kind"),
                        ["bold", "italic", "code"],
                        key=f"mk_kind_{ws.key}",
                    )
                _rt_layers = marks_in_range(
                    load_richtext(ws.key),
                    int(_mk_start),
                    int(_mk_end),
                )
                if _rt_layers:
                    st.caption(t("collab_mark_layers", layers=" + ".join(_rt_layers)))
                _layer_regions = summarize_mark_layers(ws.key)
                if _layer_regions:
                    with st.expander(t("collab_layer_map"), expanded=False):
                        for _lr in _layer_regions[:20]:
                            st.write(
                                f"{_lr['start']}:{_lr['end']} → "
                                + " + ".join(_lr.get("layers") or [])
                            )
                _mark_hist = read_mark_audit(ws.key, limit=15)
                if _mark_hist:
                    with st.expander(t("collab_mark_history"), expanded=False):
                        _mark_diffs = summarize_mark_audit_diffs(
                            _mark_hist,
                            _note_content,
                            limit=15,
                        )
                        for _md in reversed(_mark_diffs):
                            st.markdown(_md["html"], unsafe_allow_html=True)
                            if _md.get("added") or _md.get("removed") or _md.get("changed"):
                                st.caption(
                                    t(
                                        "collab_mark_diff_stats",
                                        added=_md.get("added", 0),
                                        removed=_md.get("removed", 0),
                                        changed=_md.get("changed", 0),
                                    )
                                )
                if st.button(t("collab_add_mark"), use_container_width=True, key=f"mk_add_{ws.key}"):
                    try:
                        add_mark(
                            ws.key,
                            mark=_mk_kind,
                            start=int(_mk_start),
                            end=int(_mk_end),
                            author=current_user.username if current_user else None,
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
                _c_body = st.text_input(t("collab_comment_body"), key=f"cmt_body_{ws.key}")
                if st.button(t("collab_add_comment"), use_container_width=True, key=f"cmt_add_{ws.key}"):
                    try:
                        _th = add_comment(
                            ws.key,
                            start=int(_mk_start),
                            end=int(_mk_end),
                            body=_c_body,
                            author=current_user.username if current_user else None,
                        )
                        notify_mentions(
                            ws.key,
                            _c_body,
                            from_user=current_user.username if current_user else None,
                            thread_id=_th.id,
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
                _uname = current_user.username if current_user else "local"
                _notifs = list_notifications(ws.key, _uname, unread_only=True, limit=20)
                if _notifs:
                    st.caption(t("collab_notifications"))
                    for _n in reversed(_notifs):
                        st.info(
                            t(
                                "collab_mention_notify",
                                from_user=_n.get("from_user") or "-",
                                preview=_n.get("body_preview") or "",
                            )
                        )
                    if st.button(
                        t("collab_mark_read"),
                        key=f"collab_mark_read_{ws.key}",
                        use_container_width=True,
                    ):
                        mark_notifications_read(ws.key, _uname)
                        st.rerun()
                _rt = load_richtext(ws.key)
                _cmt_hist = read_comment_audit(ws.key, limit=15)
                if _cmt_hist:
                    with st.expander(t("collab_comment_history"), expanded=False):
                        _cmt_diffs = summarize_comment_audit_diffs(
                            _cmt_hist,
                            _note_content,
                            limit=15,
                        )
                        for _cd in reversed(_cmt_diffs):
                            st.markdown(_cd["html"], unsafe_allow_html=True)
                for _th in _rt.comments:
                    _label = f"{'[✓] ' if _th.resolved else ''}{_th.author or '-'}: {_th.body[:80]}"
                    with st.expander(_label, expanded=False):
                        st.caption(f"{_th.start}:{_th.end} · {_th.id}")
                        st.markdown(render_mention_html(_th.body), unsafe_allow_html=True)
                        for _rep in _th.replies:
                            st.markdown(
                                f"— {_rep.author or '-'}: "
                                + render_mention_html(_rep.body),
                                unsafe_allow_html=True,
                            )
                        _reply = st.text_input(
                            t("collab_reply"),
                            key=f"cmt_reply_{ws.key}_{_th.id}",
                        )
                        rcol1, rcol2 = st.columns(2)
                        with rcol1:
                            if st.button(
                                t("collab_send_reply"),
                                key=f"cmt_send_{ws.key}_{_th.id}",
                                use_container_width=True,
                            ):
                                try:
                                    reply_comment(
                                        ws.key,
                                        _th.id,
                                        body=_reply,
                                        author=current_user.username if current_user else None,
                                    )
                                    notify_mentions(
                                        ws.key,
                                        _reply,
                                        from_user=current_user.username if current_user else None,
                                        thread_id=_th.id,
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.warning(str(exc))
                        with rcol2:
                            if not _th.resolved and st.button(
                                t("collab_resolve"),
                                key=f"cmt_res_{ws.key}_{_th.id}",
                                use_container_width=True,
                            ):
                                resolve_comment(ws.key, _th.id, resolved=True)
                                st.rerun()
                        _thread_hist = [
                            r for r in _cmt_hist
                            if (r.get("thread") or {}).get("id") == _th.id
                        ]
                        if _thread_hist:
                            with st.expander(t("collab_thread_diff"), expanded=False):
                                _td = summarize_comment_audit_diffs(
                                    _thread_hist,
                                    _note_content,
                                    limit=20,
                                )
                                for _tdi in reversed(_td):
                                    st.markdown(_tdi["html"], unsafe_allow_html=True)

    # Dışa aktarma mevcut oturum üzerinden (session yüklendikten sonra da çalışır)

# Session / index
emb = get_embedder(embedding_model)
if "index" not in st.session_state:
    st.session_state["index"] = load_or_create_index(
        dim=emb.dim,
        embedding_model=embedding_model,
        index_path=active_index_path,
        docstore_path=active_docstore_path,
        workspace_key=ws.key,
        backend=vector_backend(),
    )

if "chat_session" not in st.session_state:
    sid = st.session_state.get("chat_session_id")
    loaded = load_session(sid, chat_dir=active_chat_dir) if sid else None
    st.session_state["chat_session"] = loaded or create_session(chat_dir=active_chat_dir)
    st.session_state["chat_session_id"] = st.session_state["chat_session"]["id"]
if "messages" not in st.session_state:
    st.session_state["messages"] = list(st.session_state["chat_session"].get("messages") or [])

index = st.session_state["index"]
if index.embedding_model and index.embedding_model != embedding_model and index.size > 0:
    st.warning(
        "Seçili embedding modeli mevcut indeksten farklı. "
        "Doğru arama için «İndeksi Yeniden Oluştur» kullanın."
    )

if new_chat:
    created = create_session(chat_dir=active_chat_dir)
    st.session_state["chat_session"] = created
    st.session_state["chat_session_id"] = created["id"]
    st.session_state["messages"] = []
    st.rerun()

if clear_chat:
    st.session_state["messages"] = []
    st.session_state["chat_session"]["messages"] = []
    save_session(st.session_state["chat_session"], chat_dir=active_chat_dir)
    st.rerun()

if delete_chat:
    delete_session(st.session_state["chat_session_id"], chat_dir=active_chat_dir)
    created = create_session(chat_dir=active_chat_dir)
    st.session_state["chat_session"] = created
    st.session_state["chat_session_id"] = created["id"]
    st.session_state["messages"] = []
    st.rerun()

# Dosya yönetimi
st.subheader(t("documents"))
if auth_enabled():
    if ws.shared:
        st.caption("Paylaşımlı indeks: yüklenen dokümanlar tüm kullanıcıların sorgularına dahil olur.")
    else:
        st.caption("Kişisel indeks: dokümanlarınız yalnızca sizin hesabınızda görünür.")
caps = []
if ENABLE_LAYOUT_PDF:
    caps.append("Layout PDF açık: blok okuma sırası + tablolar markdown.")
if ENABLE_OCR:
    if _ocr_available():
        caps.append("OCR açık (Tesseract); zayıf metin katmanında devreye girer.")
    else:
        caps.append("OCR ayarı açık ancak Tesseract yok; yalnızca metin katmanı okunur.")
for c in caps:
    st.caption(c)

uploaded_files = None
upload_folder = ""
upload_tags = ""
if can_ingest:
    upload_folder = st.text_input(
        "Yükleme klasörü (opsiyonel)", value="", placeholder="ör. hukuk/sozlesmeler"
    )
    upload_tags = st.text_input("Etiketler (virgülle)", value="", placeholder="ör. sözleşme, 2024")
    uploaded_files = st.file_uploader(
        t("upload_files"),
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
else:
    st.info("Bu hesap yalnızca sorgu yapabilir. Doküman yükleme/silme için admin gerekir.")

col_a, col_b = st.columns(2)
with col_a:
    data_files = [
        os.path.relpath(p, active_data_dir).replace("\\", "/")
        for p in list_data_files(
            active_data_dir,
            skip_user_namespaces=ws.shared,
        )
    ]
    st.markdown("**data/**")
    if data_files:
        for name in data_files:
            st.write(f"- {name}")
    else:
        st.caption("Henüz dosya yok.")
with col_b:
    sources = index.list_sources()
    st.markdown("**İndeks kaynakları**")
    if sources:
        for name in sources:
            n_chunks = len(index.ids_for_source(name))
            # ilk chunk meta'sından klasör/etiket göster
            ids = index.ids_for_source(name)
            meta0 = index._id_to_meta.get(ids[0]) if ids else None
            extra = ""
            if meta0:
                bits = []
                if meta0.folder:
                    bits.append(f"klasör:{meta0.folder}")
                if meta0.tags:
                    bits.append("etiket:" + ",".join(meta0.tags))
                if bits:
                    extra = " — " + " · ".join(bits)
            st.write(f"- {name} ({n_chunks} chunk){extra}")
    else:
        st.caption("İndeks boş.")

delete_candidates = sorted(set(data_files) | set(sources))
if delete_candidates and can_ingest:
    to_delete = st.multiselect("Silinecek dosyalar", options=delete_candidates)
    if st.button("Seçilenleri sil", disabled=not to_delete):
        for name in to_delete:
            delete_source(name, index, active_data_dir, meta_path=active_meta_path)
        index.save(active_index_path, active_docstore_path)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
        write_audit(
            "delete_sources",
            username=current_user.username if current_user else None,
            tenant_id=ws.tenant_id,
            details={"sources": list(to_delete)},
            path=ws.audit_path,
        )
        record_metric(
            "delete",
            username=current_user.username if current_user else None,
            tenant_id=ws.tenant_id,
            values={"sources": len(to_delete)},
            path=ws.metrics_path,
        )
        st.success(f"Silindi: {', '.join(to_delete)}")
        st.rerun()

st.markdown("**Arama filtreleri**")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    source_filter = st.multiselect(
        "Dosyalar",
        options=index.list_sources(),
        default=[],
    )
with fcol2:
    folder_options = filter_folder_options(current_user, index.list_folders())
    # kök belgeleri de filtreleyebilmek için özel etiket
    root_allowed = True
    if current_user is not None:
        from rag.acl import folder_set as _folder_set

        fs = _folder_set(current_user)
        root_allowed = fs is None or "" in fs
    display_folders = folder_options + (
        ["(kök)"]
        if root_allowed
        and any(not (index._id_to_meta[i].folder or "").strip() for i in index._id_to_meta)
        else []
    )
    folder_filter_raw = st.multiselect("Klasörler", options=display_folders, default=[])
    folder_filter = [
        "" if f == "(kök)" else f for f in folder_filter_raw
    ]
    folder_filter = intersect_folder_filter(current_user, folder_filter)
with fcol3:
    tag_options = filter_tag_options(current_user, index.list_tags())
    tag_filter = st.multiselect("Etiketler", options=tag_options, default=[])
    tag_filter = intersect_tag_filter(current_user, tag_filter)
tag_mode = st.radio(
    "Etiket modu",
    options=["any", "all"],
    format_func=lambda m: "Herhangi biri" if m == "any" else "Hepsi",
    horizontal=True,
    index=0,
)

if uploaded_files:
    folder_n = normalize_folder(upload_folder)
    tags_n = normalize_tags(upload_tags)
    if not can_ingest_to(current_user, folder=folder_n, tags=tags_n):
        st.error("Bu klasör/etiket kombinasyonuna yükleme izniniz yok.")
        uploaded_files = None
if uploaded_files:
    with st.spinner("Dosyalar işleniyor..."):
        reports = []
        for uf in uploaded_files:
            save_path = ensure_data_path(uf.name, active_data_dir, folder=folder_n)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            try:
                reports.append(
                    ingest_path(
                        save_path,
                        index,
                        emb,
                        replace_existing=True,
                        folder=folder_n,
                        tags=tags_n,
                        data_dir=active_data_dir,
                        meta_path=active_meta_path,
                    )
                )
            except Exception as e:
                reports.append(
                    {
                        "source_file": uf.name,
                        "chunks_added": 0,
                        "chunks_removed": 0,
                        "skipped": True,
                        "reason": str(e),
                    }
                )
        index.save(active_index_path, active_docstore_path)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
    added = sum(r["chunks_added"] for r in reports)
    replaced = sum(1 for r in reports if r["chunks_removed"] > 0)
    skipped = sum(1 for r in reports if r.get("skipped"))
    write_audit(
        "ingest",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        details={
            "files": [r.get("source_file") for r in reports],
            "chunks_added": added,
            "replaced": replaced,
            "skipped": skipped,
            "folder": folder_n,
            "tags": tags_n,
        },
        path=ws.audit_path,
    )
    record_metric(
        "ingest",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        values={
            "files": len(reports),
            "chunks_added": added,
            "replaced": replaced,
            "skipped": skipped,
        },
        path=ws.metrics_path,
    )
    st.success(
        f"İndeks güncellendi: {added} chunk eklendi"
        + (f", {replaced} dosya yenilendi" if replaced else "")
        + (f", {skipped} dosya atlandı" if skipped else "")
        + (f" | klasör={folder_n or '(kök)'}" if folder_n is not None else "")
        + (f" | etiket={', '.join(tags_n)}" if tags_n else "")
        + "."
    )
    for r in reports:
        if r.get("skipped"):
            st.warning(f"{r['source_file']}: {r.get('reason') or 'atlandı'}")

if rebuild:
    with st.spinner("İndeks yeniden oluşturuluyor..."):
        index, reports = rebuild_from_data_dir(
            active_data_dir,
            emb,
            meta_path=active_meta_path,
            skip_user_namespaces=ws.shared,
        )
        index.save(active_index_path, active_docstore_path)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
    ok = [r for r in reports if not r.get("skipped")]
    bad = [r for r in reports if r.get("skipped")]
    write_audit(
        "rebuild",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        details={"files_ok": len(ok), "files_skipped": len(bad), "chunks": index.size},
        path=ws.audit_path,
    )
    record_metric(
        "rebuild",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        values={"files_ok": len(ok), "files_skipped": len(bad), "chunks": index.size},
        path=ws.metrics_path,
    )
    st.success(f"İndeks yeniden oluşturuldu: {len(ok)} dosya, {index.size} chunk.")
    for r in bad:
        st.warning(f"{r['source_file']}: {r.get('reason') or 'atlandı'}")

# Chat
index = st.session_state["index"]
if "bm25" not in st.session_state or st.session_state.get("bm25_index_id") != id(index):
    st.session_state["bm25"] = build_bm25_from_index(index)
    st.session_state["bm25_index_id"] = id(index)

index_empty = index.size == 0
if index_empty:
    st.info(t("index_empty"))
elif index.list_sources():
    st.caption(f"İndeksteki kaynaklar: {', '.join(index.list_sources())} ({index.size} chunk)")

with st.expander(t("compare_panel"), expanded=False):
    src_opts = index.list_sources()
    cmp_sources = st.multiselect(
        t("select_sources"),
        options=src_opts,
        default=src_opts[:2] if len(src_opts) >= 2 else src_opts,
        key="compare_sources",
    )
    cmp_focus = st.text_input(t("compare_focus"), value="", key="compare_focus")
    c_sum, c_cmp = st.columns(2)
    with c_sum:
        do_summarize = st.button(t("summarize_btn"), disabled=not cmp_sources or index_empty)
    with c_cmp:
        do_compare = st.button(
            t("compare_btn"),
            disabled=len(cmp_sources) < 2 or index_empty,
        )
    if do_summarize and cmp_sources:
        with st.spinner("Özetleniyor..."):

            def _gen(prompt: str) -> str:
                return generate_answer(provider=provider, model_name=model_name, prompt=prompt)

            try:
                summaries = summarize_sources(index, cmp_sources, _gen)
                for name, text in summaries.items():
                    st.markdown(f"**{name}**")
                    st.write(text)
            except Exception as exc:
                st.error(str(exc))
    if do_compare and len(cmp_sources) >= 2:
        with st.spinner("Karşılaştırılıyor..."):

            def _gen2(prompt: str) -> str:
                return generate_answer(provider=provider, model_name=model_name, prompt=prompt)

            try:
                result = compare_sources(
                    index,
                    cmp_sources,
                    _gen2,
                    focus=cmp_focus or "Ana noktaları ve farkları karşılaştır",
                )
                st.markdown(result)
            except Exception as exc:
                st.error(str(exc))

with st.expander("Eval paneli (retrieval smoke)"):
    st.caption(
        "Her satır: soru | beklenen_kaynak | expect_no_answer(0/1). "
        "Kaynak boş bırakılabilir."
    )
    default_eval = "örnek soru | | 1\n"
    if index.list_sources():
        default_eval = f"{index.list_sources()[0]} hakkında ne diyor? | {index.list_sources()[0]} | 0\n"
    eval_text = st.text_area("Eval seti", value=default_eval, height=120)
    if st.button("Eval çalıştır", disabled=index_empty):
        cases = []
        for line in eval_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            question = parts[0] if parts else ""
            expected = parts[1] if len(parts) > 1 and parts[1] else None
            expect_no = False
            if len(parts) > 2 and parts[2] in {"1", "true", "True", "yes"}:
                expect_no = True
            if question:
                cases.append(
                    EvalCase(
                        question=question,
                        expected_source=expected,
                        expect_no_answer=expect_no,
                    )
                )
        if not cases:
            st.warning("Geçerli eval satırı yok.")
        else:
            with st.spinner("Eval çalışıyor..."):
                results = evaluate_cases(
                    index,
                    emb,
                    cases,
                    bm25=st.session_state["bm25"],
                    top_k=top_k,
                    threshold=NO_ANSWER_THRESHOLD,
                    use_hybrid=use_hybrid,
                    alpha=hybrid_alpha,
                    use_reranker=use_reranker,
                    reranker=get_cached_reranker(use_reranker, DEFAULT_RERANKER_MODEL),
                )
            summary = summarize(results)
            st.write(
                f"Skor: {summary['passed']}/{summary['total']} "
                f"(accuracy={summary['accuracy']:.2%})"
            )
            for r in results:
                mark = "GEÇTI" if r.passed else "KALDI"
                st.markdown(f"`{mark}` **{r.question}** — {r.reason}")

# Sohbet dışa aktarma
export_session = dict(st.session_state.get("chat_session") or {})
export_session["messages"] = list(st.session_state.get("messages") or [])
ex1, ex2 = st.columns(2)
with ex1:
    st.download_button(
        t("export_json"),
        data=export_session_json(export_session),
        file_name=f"sohbet-{export_session.get('id') or 'oturum'}.json",
        mime="application/json",
        use_container_width=True,
    )
with ex2:
    st.download_button(
        t("export_md"),
        data=export_session_markdown(export_session),
        file_name=f"sohbet-{export_session.get('id') or 'oturum'}.md",
        mime="text/markdown",
        use_container_width=True,
    )

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(t("sources")):
                for src in m["sources"]:
                    st.markdown(
                        f"- [{src['label']}](#{src['anchor']}) — skor {src['score']:.3f}"
                    )

user_image = None
if ENABLE_IMAGE_OCR:
    user_image = st.file_uploader(
        t("upload_image_q"),
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="chat_image_upload",
    )

user_input = st.chat_input(t("ask_placeholder"), disabled=index_empty)

if user_input:
    t_start = time.perf_counter()
    effective_question = user_input
    image_ctx = ""
    if user_image is not None:
        try:
            vision_model = (
                DEFAULT_VISION_OPENAI_MODEL
                if provider == "openai"
                else DEFAULT_VISION_OLLAMA_MODEL
            )
            image_ctx = image_query_context(
                user_image.getvalue(),
                user_image.name,
                question=user_input,
                use_vision_llm=use_vision_llm,
                vision_provider=provider,
                vision_model=vision_model,
            )
            effective_question = merge_image_into_question(user_input, image_ctx)
            st.caption(image_ctx[:240] + ("…" if len(image_ctx) > 240 else ""))
        except Exception as exc:
            st.warning(f"Görüntü OCR/Vision atlandı: {exc}")

    user_msg = {"role": "user", "content": user_input}
    if image_ctx:
        user_msg["image_ocr"] = image_ctx[:2000]
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)
        if user_image is not None:
            st.image(user_image, caption=user_image.name, width=280)

    rewrite = rewrite_query(effective_question, mode="none")
    if rewrite_mode not in {"none", ""}:

        def _rw_gen(prompt: str) -> str:
            return generate_answer(provider=provider, model_name=model_name, prompt=prompt)

        try:
            rewrite = rewrite_query(effective_question, mode=rewrite_mode, generate_fn=_rw_gen)
            if rewrite.hypothetical:
                st.caption(f"HyDE: {rewrite.hypothetical[:180]}…")
            elif rewrite.expansions:
                st.caption("Expand: " + " | ".join(rewrite.expansions[:2]))
        except Exception as exc:
            st.warning(f"Sorgu yeniden yazma atlandı: {exc}")
            rewrite = rewrite_query(effective_question, mode="none")

    qvec = embed_rewrite(emb, rewrite)
    bm25_q = rewrite.bm25_query or effective_question
    reranker = get_cached_reranker(use_reranker, DEFAULT_RERANKER_MODEL)
    retrieved, gate_score = retrieve(
        index,
        qvec,
        bm25_q,
        bm25=st.session_state["bm25"],
        top_k=top_k,
        use_hybrid=use_hybrid,
        hybrid_alpha=hybrid_alpha,
        use_reranker=use_reranker,
        reranker=reranker,
        source_filter=source_filter or None,
        folder_filter=folder_filter or None,
        tag_filter=tag_filter or None,
        tag_mode=tag_mode,
        acl_user=current_user,
        threshold=NO_ANSWER_THRESHOLD,
    )
    if ENABLE_TABLE_BOOST:
        retrieved = prioritize_table_chunks(retrieved, question=effective_question)
    t_after_retrieval = time.perf_counter()
    no_answer = not retrieved or gate_score < NO_ANSWER_THRESHOLD

    source_payload = []
    for rc in retrieved:
        anchor = _source_anchor(rc.metadata.source_file, rc.metadata.chunk_id)
        source_payload.append(
            {
                "label": f"{rc.metadata.source_file} p.{rc.metadata.page_start}#{rc.metadata.chunk_id}",
                "anchor": anchor,
                "score": rc.score,
                "text": rc.text,
                "source_file": rc.metadata.source_file,
                "page_start": rc.metadata.page_start,
                "page_end": rc.metadata.page_end,
                "chunk_id": rc.metadata.chunk_id,
                "heading": rc.metadata.heading,
            }
        )

    with st.chat_message("assistant"):
        tool_traces = []
        tools_ctx = ""
        plan_steps = []
        memory = load_memory_from_session(st.session_state.get("chat_session"))
        if use_agent_memory and memory.context_block():
            tools_ctx = (tools_ctx + "\n" + memory.context_block()).strip()

        # Uzun vadeli profil belleği
        profile_hits = []
        if use_profile_memory:
            uname = current_user.username if current_user else "local"
            try:
                pstore = load_profile_memory(uname)
                profile_hits = search_memories(
                    pstore,
                    effective_question,
                    emb,
                    top_k=PROFILE_MEMORY_TOP_K,
                    min_score=PROFILE_MEMORY_MIN_SCORE,
                )
                if profile_hits:
                    tools_ctx = (
                        tools_ctx + "\n" + memories_context_block(profile_hits)
                    ).strip()
                    with st.expander(t("profile_hits"), expanded=False):
                        for hit in profile_hits:
                            st.write(f"- ({hit.kind}, {hit.score:.2f}) {hit.text}")
            except Exception as exc:
                st.caption(f"Profil belleği atlandı: {exc}")

        if use_agent_tools or use_agent_planner:

            def _agent_gen(prompt: str) -> str:
                return generate_answer(provider=provider, model_name=model_name, prompt=prompt)

            try:
                if use_agent_planner:
                    planned = run_planned_agent(
                        effective_question,
                        _agent_gen,
                        memory=memory if use_agent_memory else None,
                        max_steps=max(2, AGENT_MAX_STEPS),
                        use_llm_plan=True,
                    )
                    plan_steps = planned.plan
                    tool_traces = list(planned.tool_traces)
                    tools_ctx = (
                        tools_context_block(tool_traces)
                        + ("\n" + memory.context_block() if use_agent_memory else "")
                    ).strip()
                    if plan_steps:
                        with st.expander(t("agent_plan"), expanded=False):
                            for i, step in enumerate(plan_steps, 1):
                                st.write(f"{i}. {step}")
                    if tool_traces:
                        with st.expander(t("tool_traces"), expanded=False):
                            for tr in tool_traces:
                                st.write(f"**{tr.get('name')}** → {tr.get('result')}")
                elif use_agent_tools:
                    heur = run_heuristic_tools(effective_question)
                    tool_traces = list(heur.tool_traces)
                    if heur.tool_traces:
                        agent = run_agent_loop(
                            effective_question,
                            _agent_gen,
                            max_steps=max(1, AGENT_MAX_STEPS),
                            seed_tools=True,
                        )
                        tool_traces = agent.tool_traces
                        tools_ctx = (
                            tools_context_block(tool_traces)
                            + ("\n" + memory.context_block() if use_agent_memory else "")
                        ).strip()
                        with st.expander(t("tool_traces"), expanded=False):
                            for tr in tool_traces:
                                st.write(f"**{tr.get('name')}** → {tr.get('result')}")
            except Exception as exc:
                st.warning(f"Araçlar/plan atlandı: {exc}")

        if no_answer:
            if tools_ctx:
                answer = tools_ctx + "\n\n" + t("no_answer")
            else:
                answer = t("no_answer")
            st.markdown(answer)
        else:
            prompt = build_prompt(effective_question, retrieved, tools_context=tools_ctx)
            try:
                if use_stream:
                    answer = st.write_stream(
                        stream_answer(provider=provider, model_name=model_name, prompt=prompt)
                    )
                    if not isinstance(answer, str):
                        answer = "".join(answer) if answer else ""
                else:
                    with st.spinner("Yanıt üretiliyor..."):
                        answer = generate_answer(
                            provider=provider, model_name=model_name, prompt=prompt
                        )
                    st.markdown(answer)
            except Exception as e:
                answer = f"LLM çağrısı başarısız: {e}"
                st.error(answer)

            if use_source_highlight and answer and source_payload:
                highlighted = highlight_answer_html(
                    answer,
                    source_payload,
                    min_tokens=HIGHLIGHT_MIN_TOKENS,
                )
                with st.expander(t("highlighted_answer"), expanded=True):
                    st.markdown(highlighted, unsafe_allow_html=True)
                    st.caption(
                        "Sarı vurgular yanıtın kaynak chunk’larıyla örtüşen kısımlarıdır; "
                        "tıklayınca ilgili kaynağa gider."
                        if get_language() == "tr"
                        else "Yellow marks show overlap with source chunks; click to jump."
                    )

            if source_payload:
                with st.expander(t("retrieved_sources"), expanded=True):
                    mode = "hybrid" if use_hybrid else "dense"
                    if use_reranker:
                        mode += "+rerank"
                    if rewrite_mode not in {"none", ""}:
                        mode += f"+{rewrite_mode}"
                    if is_table_question(effective_question):
                        mode += "+table"
                    st.caption(f"Retrieval modu: {mode}")
                    for src in source_payload:
                        st.markdown(
                            f"<a id='{src['anchor']}'></a>",
                            unsafe_allow_html=True,
                        )
                        heading = f" | **Başlık:** {src['heading']}" if src.get("heading") else ""
                        st.markdown(
                            f"**[{src['label']}](#{src['anchor']})** — skor {src['score']:.3f}"
                            f" | sayfa {src['page_start']}–{src['page_end']}{heading}"
                        )
                        st.write(src["text"])

    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "sources": source_payload,
        "tool_traces": tool_traces if (use_agent_tools or use_agent_planner) else [],
        "plan": plan_steps,
    }
    st.session_state["messages"].append(assistant_msg)
    if use_agent_memory:
        memory = update_memory_after_turn(
            memory,
            messages=st.session_state["messages"],
            plan=plan_steps or None,
            note=None,
            generate_fn=None,
            refresh_summary=False,
        )
        st.session_state["chat_session"] = save_memory_to_session(
            dict(st.session_state["chat_session"]), memory
        )
        # Kısa bellek olgularını uzun vadeli profile yaz
        if use_profile_memory and memory.facts:
            try:
                uname = current_user.username if current_user else "local"
                pstore = load_profile_memory(uname)
                ingest_session_facts(pstore, memory.facts, embedder=emb, kind="fact")
            except Exception:
                pass
    st.session_state["chat_session"] = append_messages(
        st.session_state["chat_session"],
        [user_msg, assistant_msg],
        chat_dir=active_chat_dir,
    )
    write_audit(
        "query",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        details={
            "question": user_input[:240],
            "gate_score": round(float(gate_score), 4),
            "n_sources": len(source_payload),
            "no_answer": no_answer,
            "top_source": source_payload[0].get("source_file") if source_payload else None,
            "top_chunk": (source_payload[0].get("text") or "")[:900] if source_payload else None,
        },
        path=ws.audit_path,
    )
    t_end = time.perf_counter()
    record_metric(
        "query",
        username=current_user.username if current_user else None,
        tenant_id=ws.tenant_id,
        values={
            "gate_score": round(float(gate_score), 4),
            "n_sources": len(source_payload),
            "no_answer": no_answer,
            "latency_ms": round((t_end - t_start) * 1000, 1),
            "retrieval_ms": round((t_after_retrieval - t_start) * 1000, 1),
            "llm_ms": round((t_end - t_after_retrieval) * 1000, 1) if not no_answer else 0.0,
            "provider": provider,
            "model": model_name,
            "hybrid": use_hybrid,
            "reranker": use_reranker,
            "question": user_input[:500],
            "top_source": source_payload[0].get("source_file") if source_payload else None,
            "top_chunk": (source_payload[0].get("text") or "")[:900] if source_payload else None,
        },
        path=ws.metrics_path,
    )
    if ENABLE_DOMAIN_COLLECT and source_payload and not no_answer:
        try:
            from rag.domain_collect import append_domain_pairs, domain_pairs_path_for_tenant

            tenant_id = ws.tenant_id if ENABLE_TENANTS else None
            pairs_path = domain_pairs_path_for_tenant(tenant_id)
            append_domain_pairs(
                [
                    {
                        "anchor": user_input.strip(),
                        "positive": (source_payload[0].get("text") or "")[:900],
                        "source": "live_query",
                        "gate_score": float(gate_score),
                        "source_file": source_payload[0].get("source_file"),
                        "tenant_id": tenant_id or "global",
                    }
                ],
                pairs_path,
            )
        except Exception:
            pass
