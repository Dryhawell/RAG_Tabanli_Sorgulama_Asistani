import os
import re
import time
import streamlit as st
import requests

from rag.embed import EMBEDDING_PRESETS, Embedder, preset_for_model, resolve_embedding_model
from rag.hybrid import build_bm25_from_index
from rag.index import FaissIndex
from rag.ingest import (
    delete_source,
    ensure_data_path,
    ingest_path,
    list_data_files,
    rebuild_from_data_dir,
)
from rag.meta_store import normalize_folder, normalize_tags
from rag.eval import EvalCase, evaluate_cases, summarize
from rag.prompt import build_prompt
from rag.llm import generate_answer, stream_answer
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
)
from rag.acl import (
    can_ingest_to,
    filter_folder_options,
    filter_tag_options,
    intersect_folder_filter,
    intersect_tag_filter,
)
from rag.export_chat import export_session_json, export_session_markdown
from rag.audit import read_audit, write_audit
from rag.metrics import record_metric, summarize_metrics
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
)

st.set_page_config(page_title="RAG Not/PDF Asistanı", layout="wide")
st.title("LLM Destekli PDF / Not Sorgulama Asistanı (RAG)")

for _d in [DATA_DIR, INDEXES_DIR, METADATA_DIR, CHAT_DIR]:
    os.makedirs(_d, exist_ok=True)

# --- Auth (opsiyonel; indeks paylaşımlı, sohbetler kullanıcıya özel) ---
current_user = None
if auth_enabled():
    ensure_users_file()
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    if st.session_state["auth_user"] is None:
        st.subheader("Giriş")
        st.caption(
            "Çok kullanıcılı mod açık. Sohbet geçmişi kullanıcıya özeldir; "
            "indeks paylaşımlı veya kişisel olabilir."
        )
        if ENABLE_TENANTS:
            st.caption(f"Tenant izolasyonu açık (varsayılan tenant: `{DEFAULT_TENANT}`).")
        with st.form("login_form"):
            lu = st.text_input("Kullanıcı adı")
            lp = st.text_input("Parola", type="password")
            submitted = st.form_submit_button("Giriş yap")
        if submitted:
            user = authenticate(lu, lp)
            if user is None:
                write_audit("login_fail", username=lu, details={"reason": "invalid_credentials"})
                st.error("Geçersiz kullanıcı adı veya parola.")
            else:
                write_audit(
                    "login_success",
                    username=user.username,
                    tenant_id=user.tenant_id if ENABLE_TENANTS else None,
                )
                st.session_state["auth_user"] = user
                for k in ("chat_session", "chat_session_id", "messages", "bm25", "bm25_index_id"):
                    st.session_state.pop(k, None)
                st.rerun()
        st.info("Örnek: `users.example.json` dosyasını `metadata/users.json` olarak kopyalayın veya `RAG_AUTH_BOOTSTRAP_ADMIN` kullanın.")
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
def load_or_create_index(dim: int, embedding_model: str, index_path: str, docstore_path: str, workspace_key: str):
    # workspace_key cache ayrımı için (shared vs user:alice)
    _ = workspace_key
    if os.path.exists(index_path) and os.path.exists(docstore_path):
        try:
            loaded = FaissIndex.load(index_path, docstore_path)
            if loaded.dim == dim and (
                not loaded.embedding_model or loaded.embedding_model == embedding_model
            ):
                if not loaded.embedding_model:
                    loaded.embedding_model = embedding_model
                return loaded
        except Exception:
            pass
    return FaissIndex(dim=dim, embedding_model=embedding_model)


# Sidebar
with st.sidebar:
    st.header("Ayarlar")
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
    use_hybrid = st.checkbox("Hybrid arama (BM25 + vektör)", value=True)
    hybrid_alpha = st.slider(
        "Hybrid α (vektör ağırlığı)",
        min_value=0.0,
        max_value=1.0,
        value=float(HYBRID_ALPHA),
        step=0.05,
        disabled=not use_hybrid,
    )
    use_reranker = st.checkbox("Reranker (cross-encoder)", value=ENABLE_RERANKER)
    if use_reranker:
        st.caption(f"Model: {DEFAULT_RERANKER_MODEL}")
    use_stream = st.checkbox("Yanıtı stream et", value=True)
    rebuild = False
    if can_ingest:
        rebuild = st.button("İndeksi Yeniden Oluştur")
    else:
        st.caption("İndeks yönetimi için admin yetkisi gerekir.")

    if current_user:
        st.header("Hesap")
        st.write(f"Kullanıcı: **{current_user.username}** (`{current_user.role}`)")
        if ENABLE_TENANTS:
            st.caption(f"Tenant: `{current_user.tenant_id}`")
        if ws.shared:
            st.caption("İndeks paylaşımlı (aynı tenant içindeki kullanıcılar).")
        else:
            st.caption(f"Kişisel indeks: `{ws.data_dir}`")
        if st.button("Çıkış yap"):
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
            st.header("Admin")
            users = list_users_detail()
            st.caption(f"{len(users)} kullanıcı")
            for u in users:
                acl_f = u.get("allowed_folders")
                acl_t = u.get("allowed_tags")
                acl_txt = f" tenant={u.get('tenant_id') or DEFAULT_TENANT}"
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
                    if summary["recent"]:
                        st.markdown("**Son kayıtlar**")
                        for row in reversed(summary["recent"]):
                            vals = row.get("values") or {}
                            st.write(
                                f"`{row.get('ts')}` · **{row.get('kind')}** · "
                                f"{row.get('username') or '-'} · {vals}"
                            )

    st.header("Sohbetler")
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
        new_chat = st.button("Yeni", use_container_width=True)
    with c2:
        clear_chat = st.button("Temizle", use_container_width=True)
    with c3:
        delete_chat = st.button("Sil", use_container_width=True)

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
    )

if "chat_session" not in st.session_state:
    sid = st.session_state.get("chat_session_id")
    loaded = load_session(sid, chat_dir=active_chat_dir) if sid else None
    st.session_state["chat_session"] = loaded or create_session(chat_dir=active_chat_dir)
    st.session_state["chat_session_id"] = st.session_state["chat_session"]["id"]
if "messages" not in st.session_state:
    st.session_state["messages"] = list(st.session_state["chat_session"].get("messages") or [])

index: FaissIndex = st.session_state["index"]
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
st.subheader("Dokümanlar")
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
        "PDF veya TXT dosyaları yükleyin",
        type=["pdf", "txt"],
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
    st.info("İndeks boş. Soru sorabilmek için önce dosya yükleyin veya indeksi yeniden oluşturun.")
elif index.list_sources():
    st.caption(f"İndeksteki kaynaklar: {', '.join(index.list_sources())} ({index.size} chunk)")

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
        "Sohbeti JSON indir",
        data=export_session_json(export_session),
        file_name=f"sohbet-{export_session.get('id') or 'oturum'}.json",
        mime="application/json",
        use_container_width=True,
    )
with ex2:
    st.download_button(
        "Sohbeti Markdown indir",
        data=export_session_markdown(export_session),
        file_name=f"sohbet-{export_session.get('id') or 'oturum'}.md",
        mime="text/markdown",
        use_container_width=True,
    )

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Kaynaklar"):
                for src in m["sources"]:
                    st.markdown(
                        f"- [{src['label']}](#{src['anchor']}) — skor {src['score']:.3f}"
                    )

user_input = st.chat_input("Sorunuzu yazın...", disabled=index_empty)

if user_input:
    t_start = time.perf_counter()
    user_msg = {"role": "user", "content": user_input}
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    qvec = emb.encode([user_input])
    reranker = get_cached_reranker(use_reranker, DEFAULT_RERANKER_MODEL)
    retrieved, gate_score = retrieve(
        index,
        qvec,
        user_input,
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
        if no_answer:
            answer = "Bu bilgi dokümanda bulunmamaktadır"
            st.markdown(answer)
        else:
            prompt = build_prompt(user_input, retrieved)
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

            if source_payload:
                with st.expander("Alınan kaynaklar / chunk’lar", expanded=True):
                    mode = "hybrid" if use_hybrid else "dense"
                    if use_reranker:
                        mode += "+rerank"
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
    }
    st.session_state["messages"].append(assistant_msg)
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
        },
        path=ws.metrics_path,
    )
