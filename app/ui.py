import os
import re
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
from app.config import (
    INDEX_PATH,
    DOCSTORE_PATH,
    DATA_DIR,
    INDEXES_DIR,
    METADATA_DIR,
    CHAT_DIR,
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
def load_or_create_index(dim: int, embedding_model: str):
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCSTORE_PATH):
        try:
            loaded = FaissIndex.load(INDEX_PATH, DOCSTORE_PATH)
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
    rebuild = st.button("İndeksi Yeniden Oluştur")

    st.header("Sohbetler")
    sessions = list_sessions()
    session_ids = [s["id"] for s in sessions]
    labels = {
        s["id"]: f"{s['title']} ({s['n_messages']})"
        for s in sessions
    }

    if "chat_session_id" not in st.session_state:
        if session_ids:
            st.session_state["chat_session_id"] = session_ids[0]
        else:
            created = create_session()
            st.session_state["chat_session_id"] = created["id"]
            sessions = list_sessions()
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
        loaded = load_session(selected) or create_session()
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

# Session / index
emb = get_embedder(embedding_model)
if "index" not in st.session_state:
    st.session_state["index"] = load_or_create_index(dim=emb.dim, embedding_model=embedding_model)

if "chat_session" not in st.session_state:
    sid = st.session_state.get("chat_session_id")
    loaded = load_session(sid) if sid else None
    st.session_state["chat_session"] = loaded or create_session()
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
    created = create_session()
    st.session_state["chat_session"] = created
    st.session_state["chat_session_id"] = created["id"]
    st.session_state["messages"] = []
    st.rerun()

if clear_chat:
    st.session_state["messages"] = []
    st.session_state["chat_session"]["messages"] = []
    save_session(st.session_state["chat_session"])
    st.rerun()

if delete_chat:
    delete_session(st.session_state["chat_session_id"])
    created = create_session()
    st.session_state["chat_session"] = created
    st.session_state["chat_session_id"] = created["id"]
    st.session_state["messages"] = []
    st.rerun()

# Dosya yönetimi
st.subheader("Dokümanlar")
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
upload_folder = st.text_input("Yükleme klasörü (opsiyonel)", value="", placeholder="ör. hukuk/sozlesmeler")
upload_tags = st.text_input("Etiketler (virgülle)", value="", placeholder="ör. sözleşme, 2024")
uploaded_files = st.file_uploader(
    "PDF veya TXT dosyaları yükleyin",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns(2)
with col_a:
    data_files = [
        os.path.relpath(p, DATA_DIR).replace("\\", "/")
        for p in list_data_files(DATA_DIR)
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
if delete_candidates:
    to_delete = st.multiselect("Silinecek dosyalar", options=delete_candidates)
    if st.button("Seçilenleri sil", disabled=not to_delete):
        for name in to_delete:
            delete_source(name, index, DATA_DIR)
        index.save(INDEX_PATH, DOCSTORE_PATH)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
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
    folder_options = index.list_folders()
    # kök belgeleri de filtreleyebilmek için özel etiket
    display_folders = folder_options + (["(kök)"] if any(
        not (index._id_to_meta[i].folder or "").strip()
        for i in index._id_to_meta
    ) else [])
    folder_filter_raw = st.multiselect("Klasörler", options=display_folders, default=[])
    folder_filter = [
        "" if f == "(kök)" else f for f in folder_filter_raw
    ]
with fcol3:
    tag_filter = st.multiselect("Etiketler", options=index.list_tags(), default=[])
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
    with st.spinner("Dosyalar işleniyor..."):
        reports = []
        for uf in uploaded_files:
            save_path = ensure_data_path(uf.name, DATA_DIR, folder=folder_n)
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
        index.save(INDEX_PATH, DOCSTORE_PATH)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
    added = sum(r["chunks_added"] for r in reports)
    replaced = sum(1 for r in reports if r["chunks_removed"] > 0)
    skipped = sum(1 for r in reports if r.get("skipped"))
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
        index, reports = rebuild_from_data_dir(DATA_DIR, emb)
        index.save(INDEX_PATH, DOCSTORE_PATH)
        st.session_state["index"] = index
        st.session_state["bm25"] = build_bm25_from_index(index)
    ok = [r for r in reports if not r.get("skipped")]
    bad = [r for r in reports if r.get("skipped")]
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
        threshold=NO_ANSWER_THRESHOLD,
    )

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
        if not retrieved or gate_score < NO_ANSWER_THRESHOLD:
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
    )
