# LLM Destekli PDF / Not Sorgulama Asistanı (RAG)

## Proje Amacı
Kullanıcının yüklediği PDF veya TXT dokümanlarını analiz ederek, yalnızca bu doküman içeriklerine dayalı yanıtlar üreten bir RAG (Retrieval Augmented Generation) tabanlı asistan sunmak. Hallucination önlemek için LLM, sadece sağlanan bağlama dayanarak cevap verir ve metinde yoksa açıkça bildirir.

## Kullanılan Teknolojiler
- Python, Streamlit (UI)
- PDF okuma: PyMuPDF (pymupdf) + pdfplumber (yedek); tarama PDF için Tesseract OCR (tur+eng)
- Embedding: sentence-transformers (`all-MiniLM-L6-v2` veya `paraphrase-multilingual-MiniLM-L12-v2`)
- Vektör indeksi: FAISS (CPU, IndexFlatIP)
- Seyrek retrieval: saf Python BM25 (hybrid füzyon)
- Reranker: cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) + lexikal yedek
- LLM: OpenAI API veya lokal Ollama

## RAG Mimarisine Kısa Bakış
1. Doküman yüklenir (PDF/TXT)
2. Metin çıkarılır (gerekirse OCR); başlık/paragraf duyarlı chunk’lara bölünür
3. Chunk’lar embedding vektörlerine çevrilir
4. Vektörler FAISS’te, metinler docstore JSON’da saklanır
5. Sorgu dense → isteğe bağlı BM25 hybrid → isteğe bağlı rerank ile aranır
6. Top‑k chunk’lar katı prompt ile LLM’e verilir
7. Metinde yoksa “Bu bilgi dokümanda bulunmamaktadır” döner

## Hallucination Önleme Kuralları
- "Sadece aşağıdaki metne dayanarak cevap ver"
- "Metinde yoksa ‘Bu bilgi dokümanda bulunmamaktadır’ de"
- Skor eşiği (`RAG_NO_ANSWER_THRESHOLD`, varsayılan 0.30)
- Cevapta kaynak/chunk referansları gösterilir

## Sınırlamalar
- OCR için sistemde Tesseract gerekir (`tesseract-ocr`, `tesseract-ocr-tur`); yoksa yalnızca metin katmanı okunur
- Küçük yerel LLM modellerinde (Ollama, CPU) hız/kalite kısıtları olabilir
- Embedding modeli değişince indeks yeniden oluşturulmalıdır
- Cross-encoder reranker ilk kullanımda model indirir (CPU’da yavaş olabilir)

## Kurulum ve Çalıştırma

### Yerel (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# OCR için (opsiyonel):
# sudo apt-get install tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng
streamlit run app/ui.py
```

### LLM sağlayıcı
- **Ollama** (varsayılan): `OLLAMA_HOST` (varsayılan `http://localhost:11434`)
- **OpenAI**: `OPENAI_API_KEY` ve model (varsayılan `gpt-4o-mini`)

### CLI ingest
```bash
# data/ altındaki dosyalardan indeksi sıfırdan kur
python -m rag.cli rebuild --embedding mini-multi

# tek dosya ekle/yenile (klasör + etiket)
python -m rag.cli ingest ./data/notlar.pdf --embedding mini-en --folder hukuk --tags sozlesme,2024

# data/ listesi (alt klasörler dahil)
python -m rag.cli list
```

### Docker
```bash
docker compose up --build
# UI: http://localhost:8501
```
`data/`, `indexes/`, `metadata/` volume olarak bağlanır. İmaj Tesseract (tur/eng) içerir. Host’taki Ollama için `OLLAMA_HOST=http://host.docker.internal:11434` kullanılır.

## UI Özellikleri
- Dosya yükleme, listeleme, silme; klasör ve etiket atama
- Arama filtreleri: dosya / klasör / etiket (any|all)
- Embedding preset seçimi (EN / Multilingual)
- Hybrid arama (BM25 + vektör) ve α kaydırıcısı
- Cross-encoder reranker anahtarı
- Streaming yanıt
- Kalıcı sohbet oturumları (yeni / temizle / sil / seç)
- Eval paneli: `soru | beklenen_kaynak | expect_no_answer(0/1)`

## Ortam Değişkenleri
| Değişken | Açıklama |
|----------|----------|
| `RAG_EMBEDDING_MODEL` | Varsayılan embedding modeli |
| `RAG_TOP_K` | Top‑k |
| `RAG_NO_ANSWER_THRESHOLD` | Cosine/füzyon eşiği |
| `RAG_HYBRID_ALPHA` | Vektör ağırlığı (0–1) |
| `RAG_ENABLE_RERANKER` | Reranker varsayılanı (1/0) |
| `RAG_RERANKER_MODEL` | Cross-encoder model adı |
| `RAG_ENABLE_OCR` | OCR varsayılanı (1/0) |
| `RAG_OCR_LANGS` | Tesseract dil kodları (ör. `tur+eng`) |
| `OLLAMA_HOST` | Ollama adresi |
| `OPENAI_API_KEY` | OpenAI anahtarı |

## Dosya Yapısı
- `app/ui.py`: Streamlit arayüzü
- `app/config.py`: merkezi ayarlar
- `rag/readers.py`, `chunking.py`, `embed.py`, `index.py`
- `rag/hybrid.py`, `rag/rerank.py`, `rag/retrieve.py`
- `rag/ingest.py`, `rag/cli.py`: ingest pipeline
- `rag/meta_store.py`: kaynak klasör/etiket sidecar (`metadata/sources.json`)
- `rag/chat_store.py`: kalıcı sohbetler
- `rag/eval.py`: retrieval smoke eval
- `rag/llm.py`, `rag/prompt.py`
- `Dockerfile`, `docker-compose.yml`
- `indexes/`, `metadata/` (`chats/`, `sources.json` dahil), `data/` (alt klasörler OK): çalışma zamanı (git dışı)

## Testler
```bash
pytest -q
```
Testler model indirmez; FAISS, chunking, hybrid, retrieve, OCR mock, chat store ve CLI parser mantığını doğrular.

## Sonraki adaylar
- Çok kullanıcılı auth / paylaşımlı indeks
- Tablolar / layout-aware PDF çıkarma
- Online değerlendirme seti ve otomatik regression
