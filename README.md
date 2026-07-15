# LLM Destekli PDF / Not Sorgulama Asistanı (RAG)

## Proje Amacı
Kullanıcının yüklediği PDF veya TXT dokümanlarını analiz ederek, yalnızca bu doküman içeriklerine dayalı yanıtlar üreten bir RAG (Retrieval Augmented Generation) tabanlı asistan sunmak. Hallucination önlemek için LLM, sadece sağlanan bağlama dayanarak cevap verir ve metinde yoksa açıkça bildirir.

## Kullanılan Teknolojiler
- Python, Streamlit (UI)
- PDF okuma: PyMuPDF (pymupdf) ve pdfplumber (yedek)
- Embedding: sentence-transformers (`all-MiniLM-L6-v2` veya çok dilli `paraphrase-multilingual-MiniLM-L12-v2`)
- Vektör indeksi: FAISS (CPU, IndexFlatIP)
- Seyrek retrieval: saf Python BM25 (hybrid füzyon)
- LLM: OpenAI API veya lokal Ollama

## RAG Mimarisine Kısa Bakış
1. Doküman yüklenir (PDF/TXT)
2. Metin çıkarılır; başlık/paragraf duyarlı chunk’lara bölünür
3. Chunk’lar embedding vektörlerine çevrilir
4. Vektörler FAISS’te, metinler docstore JSON’da saklanır
5. Sorgu dense (+ isteğe bağlı BM25 hybrid) ile aranır
6. Top‑k chunk’lar katı prompt ile LLM’e verilir
7. Metinde yoksa “Bu bilgi dokümanda bulunmamaktadır” döner

## Hallucination Önleme Kuralları
- "Sadece aşağıdaki metne dayanarak cevap ver"
- "Metinde yoksa ‘Bu bilgi dokümanda bulunmamaktadır’ de"
- Skor eşiği (`RAG_NO_ANSWER_THRESHOLD`, varsayılan 0.30)
- Cevapta kaynak/chunk referansları gösterilir

## Sınırlamalar
- OCR (tarama PDF’ler) kapsam dışıdır; görüntü-tabanlı sayfalarda PyMuPDF/pdfplumber metin çıkaramayabilir
- Küçük yerel LLM modellerinde (Ollama, CPU) hız/kalite kısıtları olabilir
- Embedding modeli değişince indeks yeniden oluşturulmalıdır

## Kurulum ve Çalıştırma

### Yerel (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/ui.py
```

### LLM sağlayıcı
- **Ollama** (varsayılan): `OLLAMA_HOST` (varsayılan `http://localhost:11434`)
- **OpenAI**: `OPENAI_API_KEY` ve model (varsayılan `gpt-4o-mini`)

### CLI ingest
```bash
# data/ altındaki dosyalardan indeksi sıfırdan kur
python -m rag.cli rebuild --embedding mini-multi

# tek dosya ekle/yenile
python -m rag.cli ingest ./data/notlar.pdf --embedding mini-en

# data/ listesi
python -m rag.cli list
```

### Docker
```bash
docker compose up --build
# UI: http://localhost:8501
```
`data/`, `indexes/`, `metadata/` volume olarak bağlanır. Host’taki Ollama için `OLLAMA_HOST=http://host.docker.internal:11434` kullanılır.

## UI Özellikleri
- Dosya yükleme, listeleme, silme ve kaynak filtresi
- Embedding preset seçimi (EN / Multilingual)
- Hybrid arama (BM25 + vektör) ve α kaydırıcısı
- Streaming yanıt, sohbet temizleme
- Eval paneli: `soru | beklenen_kaynak | expect_no_answer(0/1)`

## Ortam Değişkenleri
| Değişken | Açıklama |
|----------|----------|
| `RAG_EMBEDDING_MODEL` | Varsayılan embedding modeli |
| `RAG_TOP_K` | Top‑k |
| `RAG_NO_ANSWER_THRESHOLD` | Cosine/füzyon eşiği |
| `RAG_HYBRID_ALPHA` | Vektör ağırlığı (0–1) |
| `OLLAMA_HOST` | Ollama adresi |
| `OPENAI_API_KEY` | OpenAI anahtarı |

## Dosya Yapısı
- `app/ui.py`: Streamlit arayüzü
- `app/config.py`: merkezi ayarlar
- `rag/readers.py`, `chunking.py`, `embed.py`, `index.py`
- `rag/hybrid.py`: BM25 + füzyon
- `rag/ingest.py`, `rag/cli.py`: ingest pipeline
- `rag/eval.py`: retrieval smoke eval
- `rag/llm.py`, `rag/prompt.py`
- `Dockerfile`, `docker-compose.yml`
- `indexes/`, `metadata/`, `data/`: çalışma zamanı (git dışı)

## Testler
```bash
pytest -q
```
Testler model indirmez; FAISS, chunking, hybrid, eval ve CLI parser mantığını doğrular.

## Sonraki adaylar (bilinçli olarak dışarıda bırakıldı)
- OCR hattı (ör. Tesseract / ocrmypdf)
- Çapraz kodlayıcı reranker
- Çok kullanıcılı auth / kalıcı sohbet geçmişi
