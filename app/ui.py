import os
import streamlit as st
from typing import List
import requests

from rag.embed import Embedder
from rag.index import FaissIndex
from rag.ingest import ingest_path, rebuild_from_data_dir
from rag.prompt import build_prompt
from rag.llm import generate_answer
from app.config import (
    INDEX_PATH,
    DOCSTORE_PATH,
    DATA_DIR,
    INDEXES_DIR,
    METADATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    NO_ANSWER_THRESHOLD,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
)

st.set_page_config(page_title="RAG Not/PDF Asistanı", layout="wide")
st.title("LLM Destekli PDF / Not Sorgulama Asistanı (RAG)")

# Gerekli klasörleri oluştur
for _d in [DATA_DIR, INDEXES_DIR, METADATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# Sidebar kontroller
with st.sidebar:
    st.header("Ayarlar")
    provider_options = ["ollama", "openai"]
    provider_index = provider_options.index(DEFAULT_LLM_PROVIDER) if DEFAULT_LLM_PROVIDER in provider_options else 0
    provider = st.selectbox("LLM Sağlayıcı", options=provider_options, index=provider_index)
    if provider == "ollama":
        model_name = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)
        st.caption("Öneri: küçük/quantized model (örn. phi3:mini) CPU'da daha hızlı")
        # Basit Ollama sağlık kontrolü
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
    top_k = st.slider("Top‑K", min_value=3, max_value=10, value=DEFAULT_TOP_K)
    rebuild = st.button("İndeksi Yeniden Oluştur")

@st.cache_resource(show_spinner=False)
def get_embedder():
    return Embedder(model_name=DEFAULT_EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_or_create_index(dim: int):
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCSTORE_PATH):
        try:
            return FaissIndex.load(INDEX_PATH, DOCSTORE_PATH)
        except Exception:
            pass
    return FaissIndex(dim=dim)

# Dosya yükleme
uploaded_files = st.file_uploader("PDF veya TXT dosyaları yükleyin", type=["pdf", "txt"], accept_multiple_files=True)

# Session state
if "index" not in st.session_state:
    emb = get_embedder()
    st.session_state["index"] = load_or_create_index(dim=emb.dim)

index: FaissIndex = st.session_state["index"]
emb = get_embedder()

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
    ok = [r for r in reports if not r.get("skipped")]
    bad = [r for r in reports if r.get("skipped")]
    st.success(f"İndeks yeniden oluşturuldu: {len(ok)} dosya, {index.size} chunk.")
    for r in bad:
        st.warning(f"{r['source_file']}: {r.get('reason') or 'atlandı'}")

# Chat arayüzü
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# İndeks boşsa kullanıcıyı bilgilendir
index = st.session_state["index"]
index_empty = index.size == 0
if index_empty:
    st.info("İndeks boş. Soru sorabilmek için önce dosya yükleyin veya indeksi yeniden oluşturun.")
elif index.list_sources():
    st.caption(f"İndeksteki kaynaklar: {', '.join(index.list_sources())} ({index.size} chunk)")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])# basit gösterim

user_input = st.chat_input("Sorunuzu yazın...", disabled=index_empty)

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Sorgu embedding
    qvec = emb.encode([user_input])
    retrieved = index.search(qvec, top_k=top_k)

    # Eşik kontrolü: en iyi skor düşükse doğrudan "dokümanda yok" cevabı üret
    if not retrieved or retrieved[0].score < NO_ANSWER_THRESHOLD:
        answer = "Bu bilgi dokümanda bulunmamaktadır"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        # Prompt ve LLM çağrısı
        prompt = build_prompt(user_input, retrieved)
        with st.spinner("Yanıt üretiliyor..."):
            try:
                answer = generate_answer(provider=provider, model_name=model_name, prompt=prompt)
            except Exception as e:
                answer = f"LLM çağrısı başarısız: {e}"
                st.error(answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Alınan Chunk’lar (skorlar)"):
                for rc in retrieved:
                    st.markdown(f"**Skor:** {rc.score:.3f} | **Kaynak:** {rc.metadata.source_file} | **Sayfa:** {rc.metadata.page_start} | **Chunk:** {rc.metadata.chunk_id}")
                    st.write(rc.text)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
