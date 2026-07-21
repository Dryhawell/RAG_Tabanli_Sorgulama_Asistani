# LLM Destekli PDF / Not Sorgulama Asistanı (RAG)

## Proje Amacı
Kullanıcının yüklediği PDF veya TXT dokümanlarını analiz ederek, yalnızca bu doküman içeriklerine dayalı yanıtlar üreten bir RAG (Retrieval Augmented Generation) tabanlı asistan sunmak. Hallucination önlemek için LLM, sadece sağlanan bağlama dayanarak cevap verir ve metinde yoksa açıkça bildirir.

## Kullanılan Teknolojiler
- Python, Streamlit (UI)
- PDF okuma: layout-aware (PyMuPDF blok sırası + pdfplumber tablolar → markdown); tarama PDF için Tesseract OCR (tur+eng)
- Embedding: sentence-transformers (`all-MiniLM-L6-v2` veya `paraphrase-multilingual-MiniLM-L12-v2`)
- Vektör indeksi: FAISS (CPU, IndexFlatIP)
- Seyrek retrieval: saf Python BM25 (hybrid füzyon)
- Reranker: cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) + lexikal yedek
- LLM: OpenAI API veya lokal Ollama

## RAG Mimarisine Kısa Bakış
1. Doküman yüklenir (PDF/TXT)
2. Metin çıkarılır (layout/tablo + gerekirse OCR); başlık/paragraf duyarlı chunk’lara bölünür
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

# retrieval regression (fixture + cases; hash embedder varsayılan)
python -m rag.cli eval
python -m rag.cli eval --embedding hash --output metadata/eval_report.json
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
- Opsiyonel çok kullanıcılı giriş (paylaşımlı/kişisel indeks, kullanıcıya özel sohbet)
- Opsiyonel SSO / OIDC (Authorization Code + PKCE)
- Admin paneli: kullanıcı listele / ekle / sil / ACL (klasör-etiket) / audit log / metrik dashboard
- Sohbet dışa aktarma (JSON / Markdown)
- Eval paneli: `soru | beklenen_kaynak | expect_no_answer(0/1)`

### Çok kullanıcılı mod
```bash
export RAG_ENABLE_AUTH=1
# İsteğe bağlı: ilk admin (users.json yoksa oluşturulur)
export RAG_AUTH_BOOTSTRAP_ADMIN=admin:admin
# veya:
cp users.example.json metadata/users.json
streamlit run app/ui.py
```
- İndeks varsayılan olarak **paylaşımlıdır** (`RAG_AUTH_SHARED_INDEX=1`)
- Kişisel indeks: `RAG_AUTH_SHARED_INDEX=0` → her kullanıcının `data/users/<ad>/`, `indexes/users/<ad>/`, `metadata/users/<ad>/` alanı
- Sohbetler `metadata/chats/<kullanici>/` altında ayrılır
- Paylaşımlı modda `admin` yükler/siler; `user` varsayılan yalnızca sorgu (`RAG_AUTH_USER_CAN_INGEST=1`)
- Kişisel modda her kullanıcı kendi dokümanlarını yönetebilir
- Kullanıcı ACL: `allowed_folders` / `allowed_tags` (`*` veya boş = tümü). Retrieval ve yükleme bu listeyle kısıtlanır.

### Tenant + audit
```bash
export RAG_ENABLE_AUTH=1
export RAG_ENABLE_TENANTS=1
export RAG_DEFAULT_TENANT=default
export RAG_ENABLE_AUDIT=1
```
- Tenant açıkken data/indeks/metadata kökleri `.../tenants/<tenant_id>/` altına alınır
- Kullanıcı kaydında `tenant_id` alanı; admin panelinden atanabilir
- Audit JSONL: login, logout, ingest, delete, rebuild, query, admin işlemleri (`metadata/.../audit.jsonl`)
- Metrik JSONL: sorgu gecikmesi, gate skoru, no-answer oranı, ingest/rebuild (`metadata/.../metrics.jsonl`)
- CLI: `python -m rag.cli stats` veya `python -m rag.cli stats --json`

### SSO / OIDC
```bash
export RAG_ENABLE_AUTH=1
export RAG_ENABLE_OIDC=1
export RAG_OIDC_ISSUER=https://login.example.com/realms/rag
export RAG_OIDC_CLIENT_ID=rag-assistant
export RAG_OIDC_CLIENT_SECRET=...
export RAG_OIDC_REDIRECT_URI=http://localhost:8501
# İsteğe bağlı: yalnızca SSO (parola formunu gizle)
# export RAG_OIDC_ONLY=1
# Admin grup eşlemesi (IdP groups/roles claim)
export RAG_OIDC_ADMIN_GROUPS=rag-admins,admin
streamlit run app/ui.py
```
- Authorization Code + PKCE; IdP discovery (`.well-known/openid-configuration`)
- İlk SSO girişinde kullanıcı otomatik oluşturulur (`RAG_OIDC_AUTO_PROVISION=1`)
- Rol: admin grupları veya `role` claim; tenant: `tenant_id` claim
- IdP'de redirect URI olarak Streamlit adresinizi (`http://localhost:8501`) kaydedin

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
| `RAG_ENABLE_LAYOUT_PDF` | Layout/tablo çıkarımı (1/0) |
| `RAG_ENABLE_AUTH` | Çok kullanıcılı giriş (1/0) |
| `RAG_AUTH_SHARED_INDEX` | Paylaşımlı indeks (1) / kişisel indeks (0) |
| `RAG_AUTH_USER_CAN_INGEST` | Paylaşımlı modda user yükleme yetkisi (1/0) |
| `RAG_AUTH_BOOTSTRAP_ADMIN` | `kullanici:parola` ilk admin |
| `RAG_ENABLE_TENANTS` | Tenant izolasyonu (1/0) |
| `RAG_DEFAULT_TENANT` | Varsayılan tenant kimliği |
| `RAG_ENABLE_AUDIT` | Audit log yazımı (1/0) |
| `RAG_AUDIT_LOG_PATH` | Global audit dosyası (tenant kapalıyken) |
| `RAG_ENABLE_METRICS` | Metrik kaydı (1/0) |
| `RAG_METRICS_PATH` | Global metrics dosyası (tenant kapalıyken) |
| `RAG_ENABLE_OIDC` | SSO / OIDC girişi (1/0) |
| `RAG_OIDC_ONLY` | Yalnızca SSO; parola formunu gizle (1/0) |
| `RAG_OIDC_ISSUER` | IdP issuer URL (ör. Keycloak realm) |
| `RAG_OIDC_CLIENT_ID` | OIDC client id |
| `RAG_OIDC_CLIENT_SECRET` | OIDC client secret |
| `RAG_OIDC_REDIRECT_URI` | Callback (varsayılan `http://localhost:8501`) |
| `RAG_OIDC_SCOPES` | Scope listesi |
| `RAG_OIDC_USERNAME_CLAIM` | Kullanıcı adı claim (varsayılan `preferred_username`) |
| `RAG_OIDC_TENANT_CLAIM` | Tenant claim (varsayılan `tenant_id`) |
| `RAG_OIDC_ADMIN_GROUPS` | Admin sayılacak gruplar |
| `RAG_OIDC_AUTO_PROVISION` | İlk SSO'da kullanıcı oluştur (1/0) |
| `OLLAMA_HOST` | Ollama adresi |
| `OPENAI_API_KEY` | OpenAI anahtarı |

## Dosya Yapısı
- `app/ui.py`: Streamlit arayüzü
- `app/config.py`: merkezi ayarlar
- `rag/readers.py`, `rag/pdf_layout.py`, `chunking.py`, `embed.py`, `index.py`
- `rag/hybrid.py`, `rag/rerank.py`, `rag/retrieve.py`
- `rag/ingest.py`, `rag/cli.py`: ingest pipeline
- `rag/meta_store.py`: kaynak klasör/etiket sidecar (`metadata/sources.json`)
- `rag/chat_store.py`: kalıcı sohbetler
- `rag/auth.py`, `rag/oidc.py`, `rag/workspace.py`, `rag/audit.py`, `rag/metrics.py`: auth/SSO, tenant, audit, metrikler
- `users.example.json`: örnek kullanıcı şablonu
- `rag/eval.py`, `rag/judge.py`: retrieval smoke + yanıt kalitesi judge
- `rag/llm.py`, `rag/prompt.py`
- `Dockerfile`, `docker-compose.yml`
- `indexes/`, `metadata/` (`chats/`, `users.json`, `sources.json` dahil), `data/` (alt klasörler OK): çalışma zamanı (git dışı)

## Testler
```bash
pytest -q                 # slow testler atlanır
pytest -q -m "not slow"   # CI ile aynı
# gerçek MiniLM regression (model indirir):
RUN_SLOW_EVAL=1 pytest -q -m slow
```
GitHub Actions: push/PR'da hızlı testler; Actions → CI → Run workflow ile `run_slow_eval` açılabilir.

### Eval seti
- `evals/fixtures/`: örnek TXT dokümanlar
- `evals/cases.json`: pozitif hit@k + negatif no-answer senaryoları
- CLI: `python -m rag.cli eval` (çıkış kodu 0 = min accuracy sağlandı)
- `evals/judge_cases.json`: grounded / hallucination / no-answer örnekleri
- CLI: `python -m rag.cli judge` (heuristic) veya `--mode llm --provider ollama --model phi3:mini`

## Sonraki adaylar
- Prometheus/Grafana entegrasyonu (opsiyonel dış metrik sink)
- JWKS ile id_token imza doğrulama (şu an token endpoint güvenine dayanır)
