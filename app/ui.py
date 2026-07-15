import os
import re
import streamlit as st
import requests

from rag.embed import EMBEDDING_PRESETS, Embedder, preset_for_model, resolve_embedding_model
from rag.hybrid import build_bm25_from_index, hybrid_search
from rag.index import FaissIndex
from rag.ingest import delete_source, ingest_path, list_data_files, rebuild_from_data_dir
from rag.prompt import build_prompt
from rag.llm import generate_answer, stream_answer
from app.config import (
    INDEX_PATH,
    DOCSTORE_PATH,
    DATA_DIR,
    INDEXES_DIR,
    METADATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    NO_ANSWER_THRESHOLD,
    HYBRID_ALPHA,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
)

st.set_page_config(page_title="RAG Not/PDF Asistanı", layout="wide")
st.title("LLM Destekli PDF / Not Sorgulama Asistanı (RAG)")

for _d in [DATA_DIR, INDEXES_DIR, METADATA_DIR]:
    os.makedirs(_d, exist_ok=True)


def _source_anchor(source_file: str, chunk_id: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", source_file).strip("-").lower()
    return f"src-{safe}-c{chunk_id}"


@st.cache_resource(show_spinner=True)
def get_embedder(model_name: str):
    return Embedder(model_name=model_name)


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
    use_stream = st.checkbox("Yanıtı stream et", value=True)
    rebuild = st.button("İndeksi Yeniden Oluştur")
    clear_chat = st.button("Sohbeti Temizle")

# Session / index
emb = get_embedder(embedding_model)
if "index" not in st.session_state:
    st.session_state["index"] = load_or_create_index(dim=emb.dim, embedding_model=embedding_model)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

index: FaissIndex = st.session_state["index"]
if index.embedding_model and index.embedding_model != embedding_model and index.size > 0:
    st.warning(
        "Seçili embedding modeli mevcut indeksten farklı. "
        "Doğru arama için «İndeksi Yeniden Oluştur» kullanın."
    )

if clear_chat:
    st.session_state["messages"] = []
    st.rerun()

# Dosya yönetimi
st.subheader("Dokümanlar")
uploaded_files = st.file_uploader(
    "PDF veya TXT dosyaları yükleyin",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns(2)
with col_a:
    data_files = [os.path.basename(p) for p in list_data_files(DATA_DIR)]
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
            st.write(f"- {name} ({n_chunks} chunk)")
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

source_filter = st.multiselect(
    "Aramada kullanılacak dosyalar (boş = tümü)",
    options=index.list_sources(),
    default=[],
)

if uploaded_files:
    with st.spinner("Dosyalar işleniyor..."):
        reports = []
        for uf in uploaded_files:
            save_path = os.path.join(DATA_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            try:
                reports.append(ingest_path(save_path, index, emb, replace_existing=True))
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
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    qvec = emb.encode([user_input])
    dense_hits = index.search(qvec, top_k=max(top_k * 3, top_k))
    if use_hybrid:
        retrieved = hybrid_search(
            index,
            qvec,
            user_input,
            st.session_state["bm25"],
            top_k=max(top_k * 3, top_k),
            alpha=hybrid_alpha,
        )
    else:
        retrieved = dense_hits

    if source_filter:
        retrieved = [rc for rc in retrieved if rc.metadata.source_file in source_filter]
        dense_hits = [rc for rc in dense_hits if rc.metadata.source_file in source_filter]

    retrieved = retrieved[:top_k]
    dense_hits = dense_hits[:top_k]

    if use_hybrid:
        best_dense = dense_hits[0].score if dense_hits else 0.0
        best_fused = retrieved[0].score if retrieved else 0.0
        gate_score = max(best_dense, best_fused)
    else:
        gate_score = dense_hits[0].score if dense_hits else 0.0

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

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_payload,
        }
    )
