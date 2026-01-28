import os
import streamlit as st
from typing import List
import requests

from rag.readers import read_document
from rag.chunking import chunk_pages
from rag.embed import Embedder
from rag.index import FaissIndex
from rag.prompt import build_prompt
from rag.llm import generate_answer
from app.config import INDEX_PATH, DOCSTORE_PATH, DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, NO_ANSWER_THRESHOLD

st.set_page_config(page_title="RAG Not/PDF Asistanı", layout="wide")
st.title("LLM Destekli PDF / Not Sorgulama Asistanı (RAG)")

# Gerekli klasörleri oluştur
for _d in ["data", "indexes", "metadata"]:
    os.makedirs(_d, exist_ok=True)

# Sidebar kontroller
with st.sidebar:
    st.header("Ayarlar")
    provider = st.selectbox("LLM Sağlayıcı", options=["ollama", "openai"], index=0)
    if provider == "ollama":
        model_name = st.text_input("Ollama Model", value="phi3:mini")
        st.caption("Öneri: küçük/quantized model (örn. phi3:mini) CPU'da daha hızlı")
        # Basit Ollama sağlık kontrolü
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                st.success("Ollama çalışıyor (localhost:11434)")
            else:
                st.warning("Ollama'a ulaşılamadı veya beklenmeyen yanıt.")
        except Exception:
            st.warning("Ollama kapalı görünüyor. Lütfen Ollama'yı başlatın.")
    else:
        model_name = st.text_input("OpenAI Model", value="gpt-3.5-turbo")
        st.caption("OPENAI_API_KEY çevre değişkeni gerekli")
        if not os.getenv("OPENAI_API_KEY"):
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
        for uf in uploaded_files:
            save_path = os.path.join("data", uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            source_name, pages = read_document(save_path)
            chunk_texts, metas = chunk_pages(source_file=source_name, pages=pages)
            if not chunk_texts:
                continue
            vecs = emb.encode(chunk_texts)
            index.add(vecs, chunk_texts, metas)
        # Kalıcılık
        index.save(INDEX_PATH, DOCSTORE_PATH)
    st.success("İndeks güncellendi.")

if rebuild:
    # Basit yeniden yükleme: docstore'u yeniden okur ve FAISS'i sıfırlar
    st.session_state["index"] = FaissIndex(dim=emb.dim)
    index = st.session_state["index"]
    # Var olan data klasöründeki dosyaları tekrar işleme
    files = [os.path.join("data", x) for x in os.listdir("data")]
    with st.spinner("İndeks yeniden oluşturuluyor..."):
        for path in files:
            try:
                source_name, pages = read_document(path)
                chunk_texts, metas = chunk_pages(source_file=source_name, pages=pages)
                if not chunk_texts:
                    continue
                vecs = emb.encode(chunk_texts)
                index.add(vecs, chunk_texts, metas)
            except Exception:
                continue
        index.save(INDEX_PATH, DOCSTORE_PATH)
    st.success("İndeks yeniden oluşturuldu.")

# Chat arayüzü
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# İndeks boşsa kullanıcıyı bilgilendir
index_empty = getattr(index.idmap, "ntotal", 0) == 0
if index_empty:
    st.info("İndeks boş. Soru sorabilmek için önce dosya yükleyin veya indeksi yeniden oluşturun.")

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
