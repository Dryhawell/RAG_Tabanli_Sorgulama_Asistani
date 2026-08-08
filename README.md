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
- Sidebar dil seçimi (TR / EN)
- Sorgu yeniden yazma: HyDE / genişletme
- Doküman karşılaştırma ve özet paneli
- Agentic araçlar: hesap makinesi, takvim, web arama
- Yanıtta kaynak chunk vurgulama (HTML `<mark>`)
- Multimodal: görüntü OCR + tablo sorusu boost; agent bellek/planlama
- Vision-LLM (GPT-4o / LLaVA) ve uzun vadeli vektör profil belleği
- Paylaşım linkleri (salt okunur sohbet) ve işbirlikçi not
- Domain / fine-tuned embedding preset
- WebSocket çoklu düzenleyici + embedding fine-tuning pipeline
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
- `RAG_OIDC_VERIFY_JWKS=1` (varsayılan): id_token JWKS ile doğrulanır

### Prometheus
```bash
export RAG_ENABLE_METRICS=1
export RAG_ENABLE_PROMETHEUS=1
export RAG_PROMETHEUS_PORT=9108
# Ayrı süreç:
python -m rag.cli prometheus
# veya UI açıkken aynı process içinde /metrics dinlenir
# Scrape: http://localhost:9108/metrics
# Dump (sunucusuz):
python -m rag.cli prometheus --dump
```
Grafana'da Prometheus datasource ekleyip `rag_queries_total`, `rag_query_latency_seconds` panelleri kurulabilir.
Judge soft-fail paneli: `grafana/dashboards/rag_judge.json` (provisioning: `grafana/provisioning/dashboards/`).
Judge Slack alert kuralı: `grafana/alerting/rag_judge_soft_fail.yaml` + CI `scripts/ci_judge_slack_alert.py` (`RAG_JUDGE_SLACK_WEBHOOK`, opsiyonel `RAG_JUDGE_PAGERDUTY_ROUTING_KEY` / `RAG_JUDGE_OPSGENIE_API_KEY`; soft-fail → OK geçişinde auto-resolve, state: `RAG_JUDGE_ALERT_STATE`; manuel ack: `POST /judge/ack` + `RAG_JUDGE_ACK_TOKEN` veya admin UI).
CI judge state: artifact + `actions/cache` (`metadata/judge_alert_state.json`) ile run'lar arası kalıcılık; opsiyonel harici store: `RAG_JUDGE_ALERT_STATE_URL`.
Slack ack deep-link: `RAG_JUDGE_ACK_PUBLIC_URL` → `/judge/ack-form`; interactive + ephemeral + rate limit; audit: `judge-ack-export` / `judge-ack-purge` / `judge-ack-digest` (Block Kit re-export → Slack `files.upload`; quiet override: `RAG_JUDGE_ACK_DIGEST_QUIET_HOURS_JSON`).
Observability stack: `docker compose --profile obs up -d` (Prometheus + Alertmanager + Tempo + Grafana).
Alertmanager: rotate/reload/silence/inhibit + `--check-config`; CI: `--tune-equal` / `--diff-inhibit` / main `--apply-equal` (amtool gate + commit).
LLM cost recording rules: `grafana/rules/rag_llm_cost.yml` → `rag:llm_cost_usd_per_hour`.
Vektör migrasyon: `python -m rag.cli migrate-vector --source faiss --target qdrant` (dual-write: `RAG_VECTOR_DUAL_WRITE=qdrant`).
Dual-write: burn-rate webhook `POST /hooks/dual-write-catch-up` → in-process catch-up; fail/`not_dual_write` → GitHub `workflow_dispatch` fallback (`GH_PAT`).
VAPID üretimi: `python -m rag.cli collab-notifications --generate-vapid`
VAPID rotate/vault: `python -m rag.cli collab-notifications --rotate-vapid` (Actions: **VAPID Rotate**; secret sync: `GH_PAT` + `--update-github-secrets` + opsiyonel `--vapid-github-environment`)
Digest alert: `python -m rag.cli collab-notifications --digest-alert-check --digest-alert-all` (cron: `.github/workflows/digest-alert.yml`; tenant webhook: `RAG_DIGEST_ALERT_WEBHOOKS_JSON` veya `metadata/digest_alert_webhooks.json`)
OpenTelemetry (opsiyonel): `RAG_OTEL_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT` + `opentelemetry-*` paketleri; resource: `OTEL_SERVICE_NAME`, `RAG_ENVIRONMENT`, `OTEL_RESOURCE_ATTRIBUTES` (`rag/otel.py`; spans: retrieve/ingest/judge/llm + token attrs).
Prometheus: `rag_llm_tokens_total` / query latency exemplars (Grafana Tempo link).
Delta rebuild: `python -m rag.cli rebuild --delta` (fingerprint + seçici `chunk_uid` re-embed).

### Qdrant (uzak / dağıtık vektör DB)
```bash
# Yerel gömülü (sunucusuz):
export RAG_VECTOR_BACKEND=qdrant
export RAG_QDRANT_PATH=indexes/qdrant_local
# veya uzak:
# export RAG_QDRANT_URL=http://localhost:6333
# export RAG_QDRANT_COLLECTION=rag_chunks
streamlit run app/ui.py

# Docker ile Qdrant:
docker compose --profile qdrant up -d qdrant
export RAG_VECTOR_BACKEND=qdrant RAG_QDRANT_URL=http://localhost:6333
```
FAISS varsayılandır. Qdrant açıkken hybrid BM25 hâlâ yerel docstore üzerinden çalışır.

### UI dili
Sidebar'dan **Dil / Language** seçin (`tr` / `en`) veya `RAG_UI_LANG=en`.

### HyDE / sorgu yeniden yazma
Sidebar → **Sorgu yeniden yazma**: `HyDE`, `Genişlet` veya ikisi.
```bash
export RAG_ENABLE_QUERY_REWRITE=1
export RAG_QUERY_REWRITE_MODE=hyde   # none | hyde | expand | hyde+expand
```
HyDE: LLM hipotetik paragraf üretir, embedding ile aranır. Expand: alternatif soru ifadeleri BM25/vektöre eklenir.

### Doküman karşılaştır / özetle
UI'da **Doküman karşılaştır / özetle** panelinden birden fazla kaynak seçip Özetle veya Karşılaştır.

### Araçlar (agent) + kaynak vurgulama
```bash
export RAG_ENABLE_AGENT_TOOLS=1
export RAG_ENABLE_SOURCE_HIGHLIGHT=1
streamlit run app/ui.py
```
- Sidebar’dan araçları açın: hesap makinesi, takvim, DuckDuckGo web arama
- Yanıt sonrası sarı vurgular kaynak chunk örtüşmelerini gösterir; tıklayınca ilgili kaynağa gider

### Multimodal + bellek/plan
```bash
export RAG_ENABLE_IMAGE_OCR=1
export RAG_ENABLE_TABLE_BOOST=1
export RAG_ENABLE_AGENT_MEMORY=1
export RAG_ENABLE_AGENT_PLANNER=1
```
- PNG/JPG indekslenebilir (OCR → metin chunk)
- Sohbette görüntü yükleyip soru sorabilirsiniz
- Tablo sorularında `[Tablo]` chunk’ları öne alınır
- Agent belleği oturumda saklanır; planlı agent adım adım TOOL_CALL üretir

### Vision-LLM + uzun vadeli profil belleği
```bash
# OpenAI:
export RAG_ENABLE_VISION_LLM=1
export RAG_VISION_OPENAI_MODEL=gpt-4o-mini
export OPENAI_API_KEY=...
# veya Ollama LLaVA:
# export RAG_LLM_PROVIDER=ollama
# export RAG_VISION_OLLAMA_MODEL=llava

export RAG_ENABLE_PROFILE_MEMORY=1
streamlit run app/ui.py
```
- Vision açıksa görüntü GPT-4o/LLaVA ile yorumlanır; OCR yedek kalır
- Profil belleği: `metadata/profiles/<user>/long_memory.json` (embedding araması)
- Oturum olguları otomatik uzun vadeli belleğe taşınır

### Paylaşım linkleri + işbirlikçi not
```bash
export RAG_ENABLE_SHARE_LINKS=1
export RAG_PUBLIC_BASE_URL=http://localhost:8501
export RAG_SHARE_LINK_TTL_DAYS=7
export RAG_ENABLE_COLLAB_NOTES=1
streamlit run app/ui.py
```
- Sidebar’dan salt okunur sohbet paylaşım linki oluşturun (`?share=<token>`)
- Link giriş gerektirmez; süre dolunca veya iptal edilince geçersiz olur
- İşbirlikçi not: çalışma alanına özel paylaşımlı metin; revizyon geçmişi ve çakışma uyarısı

### Domain / fine-tuned embedding
```bash
export RAG_DOMAIN_EMBEDDING_MODEL=/path/to/fine-tuned-model
# veya HuggingFace model adı:
# export RAG_DOMAIN_EMBEDDING_MODEL=my-org/legal-miniLM
export RAG_ENABLE_DOMAIN_EMBEDDING=1
streamlit run app/ui.py
```
- Embedding preset listesine **Domain / fine-tuned** eklenir
- Model değişince indeksi yeniden oluşturun

### WebSocket çoklu düzenleyici
```bash
export RAG_ENABLE_COLLAB_WS=1
export RAG_COLLAB_WS_PORT=8765
# Ayrı süreç:
python -m rag.cli collab-serve --port 8765
# veya UI açıkken aynı process içinde ws://localhost:8765
```
- İşbirlikçi not: `join` → `edit` → `sync` / `conflict`
- Örnek mesaj: `{"op":"join","workspace_key":"tenant:default|shared","username":"alice"}`

### Embedding fine-tuning pipeline
```bash
python -m rag.cli embed-pairs --output metadata/embed_pairs.jsonl
python -m rag.cli embed-train --embedding mini-multi --pairs metadata/embed_pairs.jsonl --epochs 2
python -m rag.cli embed-eval --embedding mini-multi --finetuned models/embed-finetuned
# Tam döngü (pairs + train + eval):
python -m rag.cli embed-pipeline --embedding mini-en --output models/ci-embed-finetuned
export RAG_DOMAIN_EMBEDDING_MODEL=models/embed-finetuned
export RAG_ENABLE_DOMAIN_EMBEDDING=1
```
- GitHub Actions: `embed-finetune.yml` (haftalık schedule + workflow_dispatch)
- CI fast job: `embed-pipeline --pairs-only` smoke

### Otomatik domain veri toplama
```bash
export RAG_ENABLE_DOMAIN_COLLECT=1
python -m rag.cli domain-collect --output metadata/domain_training/pairs.jsonl
python -m rag.cli federated-pool --output metadata/federated/training_pool.jsonl
python -m rag.cli privacy-pool --noise 1.0 --clip 1.0
python -m rag.cli embed-pipeline --collected-pairs metadata/domain_training/pairs.jsonl --federated-pool metadata/federated/training_pool.jsonl --private-pool metadata/federated/private_training_pool.jsonl
```
- Sohbet / audit / metriklerden (soru, chunk) çiftleri; tenant bazlı `metadata/tenants/<id>/domain_training/`
- Federated havuz: tüm tenant çiftlerini tek JSONL'de birleştirir
- **Privacy pool**: DP-SGD lite (clip + Gaussian) + secure aggregation PoC
- **DP-SGD eğitim**: `python -m rag.cli dp-train --no-opacus` (Opacus varsa `--opacus`)
- **ST + Opacus**: `python -m rag.cli st-dp-train --embedding mini-en` (frozen backbone + DP head)
- **LoRA + DP**: `python -m rag.cli lora-dp-train --mock` (CI) veya `--no-mock --production --opacus`
- **LoRA eval**: `python -m rag.cli lora-dp-eval --lora-dir models/lora-dp-embed --min-accuracy 0.8 --min-delta 0.0 --per-case-output metadata/lora_per_case.json`
- **@mention**: yorumlarda `@kullanici` → bildirim + WebSocket `mention_notify`
- **Bildirim merkezi**: `python -m rag.cli collab-notifications --user alice`
- **E-posta/webhook**: `RAG_NOTIFY_SMTP_HOST`, `RAG_NOTIFY_WEBHOOK_URL` (Slack/Discord/generic)
- **Digest**: `python -m rag.cli collab-notifications --digest --user alice --digest-mentions-only --digest-group-by thread` (e-posta + Slack/Discord/Teams + mobil push)
- **Quiet hours**: profil + thread reply + flush; **kanal politikası** `quiet_channels`
- **Digest kanalları**: `digest_channels` matrisi (email/webhook/push)
- **Mobil push PoC**: FCM/APNs invalid-token revoke + metrik; günlük prune workflow
- **LoRA rollback**: CI status + PR comment bot
- **Collab presence**: disk snapshot + typing
- **Judge CI**: heuristic gate; Actions → `run_judge_llm` (OpenAI/Ollama secrets)
- GitHub Actions: digest flush + push prune
- **Mark katmanları**: çoklu stil birleşimi (bold+italic) canlı editör + katman haritası + audit diff görselleştirme

### CRDT işbirlikçi not
```bash
export RAG_ENABLE_COLLAB_CRDT=1
export RAG_ENABLE_COLLAB_WS=1
export RAG_ENABLE_COLLAB_LIVE_EDITOR=1
export RAG_ENABLE_COLLAB_RICHTEXT=1
```
- RGA-tarzı CRDT: eşzamanlı düzenlemeler otomatik birleşir (`rag/collab_crdt.py`)
- **Canlı düzenleyici**: contenteditable + görsel remote imleç/selection overlay
- Undo/redo: Ctrl+Z / Ctrl+Y (WebSocket `undo`/`redo`) + UI butonları
- **Paste / IME**: composition sırasında sync kapalı; paste `paste` op; IME `compositionend` → full commit
- **Rich-text**: kalın/italik/kod işaretleri + yorum thread'leri; canlı editörde inline HTML
- WebSocket: `edit` / `paste` / `ime_commit` / `cursor` / presence / undo / redo / `rich_ops` / `notify_list`
- CRDT düzenleme sonrası rich-text mark/yorum aralıkları otomatik yeniden eşlenir

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
| `RAG_OIDC_VERIFY_JWKS` | id_token JWKS imza doğrulama (1/0, varsayılan 1) |
| `RAG_ENABLE_PROMETHEUS` | Prometheus sink (1/0) |
| `RAG_PROMETHEUS_PORT` | /metrics portu (varsayılan 9108) |
| `RAG_PROMETHEUS_ADDR` | Bind adresi (varsayılan 0.0.0.0) |
| `RAG_VECTOR_BACKEND` | `faiss` (varsayılan) veya `qdrant` |
| `RAG_QDRANT_URL` | Uzak Qdrant HTTP URL |
| `RAG_QDRANT_PATH` | Yerel gömülü Qdrant dizini |
| `RAG_QDRANT_COLLECTION` | Collection adı |
| `RAG_QDRANT_API_KEY` | Opsiyonel API anahtarı |
| `RAG_UI_LANG` | UI dili (`tr` / `en`) |
| `RAG_ENABLE_QUERY_REWRITE` | HyDE/expand varsayılanını aç (1/0) |
| `RAG_QUERY_REWRITE_MODE` | `none` / `hyde` / `expand` / `hyde+expand` |
| `RAG_ENABLE_AGENT_TOOLS` | Hesap/takvim/web araçları (1/0) |
| `RAG_AGENT_MAX_STEPS` | Agent TOOL_CALL döngü üst sınırı |
| `RAG_ENABLE_SOURCE_HIGHLIGHT` | Yanıtta kaynak vurgulama (1/0) |
| `RAG_HIGHLIGHT_MIN_TOKENS` | Vurgu için min ortak token |
| `RAG_ENABLE_IMAGE_OCR` | Görüntü OCR (1/0) |
| `RAG_ENABLE_TABLE_BOOST` | Tablo sorularında chunk önceliği (1/0) |
| `RAG_ENABLE_AGENT_MEMORY` | Oturum agent belleği (1/0) |
| `RAG_ENABLE_AGENT_PLANNER` | Çok adımlı planlı agent (1/0) |
| `RAG_ENABLE_VISION_LLM` | GPT-4o / LLaVA görüntü anlama (1/0) |
| `RAG_VISION_OPENAI_MODEL` | Vision OpenAI modeli |
| `RAG_VISION_OLLAMA_MODEL` | Vision Ollama modeli (llava) |
| `RAG_ENABLE_PROFILE_MEMORY` | Uzun vadeli vektör bellek (1/0) |
| `RAG_PROFILE_MEMORY_TOP_K` | Profil bellek retrieval k |
| `RAG_PROFILE_MEMORY_MIN_SCORE` | Profil bellek skor eşiği |
| `RAG_ENABLE_SHARE_LINKS` | Sohbet paylaşım linkleri (1/0) |
| `RAG_SHARE_LINK_TTL_DAYS` | Paylaşım linki geçerlilik süresi (gün) |
| `RAG_PUBLIC_BASE_URL` | Paylaşım URL tabanı |
| `RAG_ENABLE_COLLAB_NOTES` | İşbirlikçi paylaşımlı not (1/0) |
| `RAG_ENABLE_COLLAB_WS` | İşbirlikçi WebSocket sunucusu (1/0) |
| `RAG_COLLAB_WS_PORT` | WebSocket portu (varsayılan 8765) |
| `RAG_COLLAB_WS_PUBLIC_HOST` | UI’da gösterilen WS host |
| `RAG_ENABLE_COLLAB_CRDT` | CRDT birleştirme (1/0, varsayılan açık) |
| `RAG_ENABLE_COLLAB_LIVE_EDITOR` | Canlı WebSocket düzenleyici (1/0) |
| `RAG_ENABLE_COLLAB_RICHTEXT` | Rich-text marks + yorum thread'leri (1/0) |
| `RAG_ENABLE_DOMAIN_COLLECT` | Canlı sorgulardan domain çift toplama (1/0) |
| `RAG_DOMAIN_PAIRS_PATH` | Toplanan domain çiftleri JSONL |
| `RAG_DOMAIN_COLLECT_MIN_GATE` | Toplama için min gate skoru |
| `RAG_FEDERATED_POOL_PATH` | Federated tenant eğitim havuzu JSONL |
| `RAG_FEDERATED_MIN_PER_TENANT` | Havuza dahil min çift / tenant |
| `RAG_ENABLE_FEDERATED_PRIVACY` | DP privacy pool (1/0) |
| `RAG_FEDERATED_DP_NOISE` | DP Gaussian noise multiplier |
| `RAG_FEDERATED_DP_CLIP` | DP L2 clip norm |
| `RAG_FEDERATED_SECRET` | Secure aggregation secret |
| `RAG_PRIVATE_FEDERATED_POOL_PATH` | Private federated pool çıktısı |
| `RAG_ENABLE_DP_TRAIN` | DP-SGD eğitim (1/0) |
| `RAG_DP_TRAIN_OUTPUT_DIR` | DP model çıktı dizini |
| `RAG_DP_TRAIN_NOISE` | DP-SGD noise multiplier |
| `RAG_DP_TRAIN_MAX_GRAD_NORM` | DP-SGD max grad norm |
| `RAG_DP_TRAIN_DELTA` | DP delta |
| `RAG_DP_TRAIN_USE_OPACUS` | Opacus PrivacyEngine (1/0) |
| `RAG_ST_DP_TRAIN_OUTPUT_DIR` | ST+DP model çıktı dizini |
| `RAG_ST_DP_HEAD_DIM` | ST DP projeksiyon boyutu |
| `RAG_ST_DP_MAX_SEQ_LENGTH` | ST max seq length |
| `RAG_ST_DP_FREEZE_BACKBONE` | Omurgayı dondur (1/0) |
| `RAG_LORA_DP_TRAIN_OUTPUT_DIR` | LoRA+DP model çıktı dizini |
| `RAG_LORA_DP_RANK` | LoRA rank |
| `RAG_LORA_DP_ALPHA` | LoRA alpha |
| `RAG_LORA_DP_MOCK` | Mock LoRA yolu (1/0) |
| `RAG_LORA_DP_OPACUS_PRODUCTION` | Opacus ModuleValidator + üretim grad_sample (1/0) |
| `RAG_LORA_DP_SECURE_MODE` | Opacus secure RNG (1/0) |
| `RAG_LORA_DP_GRAD_SAMPLE_MODE` | Opacus grad_sample_mode (varsayılan hooks) |
| `RAG_LORA_DP_EVAL_MIN_ACCURACY` | LoRA eval CI regression eşiği (varsayılan 0.0) |
| `RAG_LORA_DP_EVAL_MIN_DELTA` | Base'e göre min delta (varsayılan 0.0; negatif = düşüş toleransı) |
| `RAG_ENABLE_COLLAB_NOTIFY_DISPATCH` | E-posta/webhook bildirim dağıtımı (1/0) |
| `RAG_NOTIFY_SMTP_HOST` | SMTP sunucusu (boş = e-posta kapalı) |
| `RAG_NOTIFY_SMTP_PORT` | SMTP portu |
| `RAG_NOTIFY_SMTP_USER` | SMTP kullanıcı |
| `RAG_NOTIFY_SMTP_PASSWORD` | SMTP parola |
| `RAG_NOTIFY_FROM_EMAIL` | Gönderen e-posta |
| `RAG_NOTIFY_WEBHOOK_URL` | Push/webhook URL (Slack/Discord) |
| `RAG_NOTIFY_DIGEST_HOURS` | Digest özet penceresi (saat, varsayılan 24) |
| `RAG_NOTIFY_DIGEST_MENTIONS_ONLY` | Digest yalnızca @mention (1/0) |
| `RAG_NOTIFY_DIGEST_GROUP_BY` | Digest gruplama: `thread` / `workspace` / `none` |
| `RAG_NOTIFY_DIGEST_MIN_PER_WORKSPACE` | Workspace min bildirim eşiği (varsayılan 1) |
| `RAG_NOTIFY_DIGEST_HTML` | Digest e-posta HTML şablonu (1/0, varsayılan açık) |
| `RAG_NOTIFY_DIGEST_QUIET_HOURS` | Quiet hours yerel `HH:MM-HH:MM` (örn. `22:00-07:00`, boş=kapalı) |
| `RAG_NOTIFY_DIGEST_TIMEZONE` | Quiet hours IANA timezone (varsayılan `UTC`) |
| `RAG_NOTIFY_TENANT_TIMEZONES` | Tenant→timezone (`default:Europe/Istanbul,acme:America/New_York`) |
| `RAG_NOTIFY_DIGEST_THREAD_REPLY` | Quiet hours'ta Slack/Teams thread özeti (1/0, varsayılan açık) |
| `RAG_NOTIFY_DIGEST_SLACK_THREAD_TS` | Slack parent `thread_ts` (yoksa state dosyası) |
| `RAG_NOTIFY_DIGEST_TEAMS_REPLY_ID` | Teams `replyToId` |
| `RAG_NOTIFY_SLACK_BOT_TOKEN` | Slack `chat.postMessage` bot token (opsiyonel) |
| `RAG_NOTIFY_SLACK_CHANNEL` | Slack kanal id (bot path) |
| `RAG_NOTIFY_PUSH_URL` | Mobil push endpoint (FCM/APNs gateway / generic) |
| `RAG_NOTIFY_PUSH_API_KEY` | Push API anahtarı / Bearer token |
| `RAG_NOTIFY_PUSH_PROVIDER` | `generic` / `fcm` / `apns` |
| `RAG_NOTIFY_PUSH_TOKEN_TTL_DAYS` | Push token TTL (gün, varsayılan 90) |
| `RAG_NOTIFY_PUSH_FCM_PROJECT_ID` | FCM v1 project id (URL boşsa endpoint üretir) |
| `RAG_NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON` | Service account JSON yolu veya inline JSON |
| `RAG_NOTIFY_PUSH_APNS_KEY_ID` | APNs Key ID |
| `RAG_NOTIFY_PUSH_APNS_TEAM_ID` | Apple Team ID |
| `RAG_NOTIFY_PUSH_APNS_TOPIC` | Bundle ID / apns-topic |
| `RAG_NOTIFY_PUSH_APNS_P8_PATH` | `.p8` dosya yolu |
| `RAG_NOTIFY_PUSH_APNS_P8_CONTENT` | `.p8` PEM içeriği (inline) |
| `RAG_NOTIFY_PUSH_APNS_USE_SANDBOX` | APNs sandbox host (1/0) |
| `RAG_COLLAB_PRESENCE_TTL_SEC` | Presence snapshot TTL (saniye, varsayılan 90) |
| `RAG_JUDGE_MIN_ACCURACY` | Judge CI gate eşiği (varsayılan 1.0) |
| `RAG_JUDGE_MODE` | `heuristic` / `llm` (CI varsayılan heuristic) |
| `RAG_LORA_DP_EVAL_AUTO_ROLLBACK` | Gate başarısızsa rollback_suggestion.json yaz (1/0) |
| `RAG_EMBED_PIPELINE_REPORT_PATH` | Pipeline JSON rapor yolu |
| `RAG_DOMAIN_EMBEDDING_MODEL` | Domain/fine-tuned embedding model yolu veya Hub adı |
| `RAG_ENABLE_DOMAIN_EMBEDDING` | Domain embedding varsayılan preset (1/0) |
| `OLLAMA_HOST` | Ollama adresi |
| `OPENAI_API_KEY` | OpenAI anahtarı |

## Dosya Yapısı
- `app/ui.py`: Streamlit arayüzü
- `app/config.py`: merkezi ayarlar
- `rag/index.py`, `rag/qdrant_index.py`, `rag/store.py`: FAISS / Qdrant vektör deposu
- `rag/i18n.py`: UI lokalizasyon (TR/EN)
- `rag/query_rewrite.py`, `rag/compare.py`: HyDE/expand ve çoklu doküman özet/karşılaştırma
- `rag/tools.py`, `rag/agent.py`, `rag/highlight.py`: araçlar, agent döngüsü, kaynak vurgulama
- `rag/vision.py`, `rag/vision_llm.py`, `rag/memory.py`, `rag/planner.py`, `rag/profile_memory.py`
- `rag/share_links.py`, `rag/collab_notes.py`, `rag/collab_ws.py`, `rag/collab_crdt.py`, `rag/collab_component.py`
- `rag/embed_finetune.py`, `rag/domain_collect.py`, `rag/federated_pool.py`, `rag/privacy_federated.py`, `rag/dp_train.py`, `rag/st_dp_train.py`, `rag/st_lora_dp.py`, `rag/lora_dp_opacus.py`, `rag/lora_dp_eval.py`, `rag/collab_presence.py`, `rag/collab_undo.py`, `rag/collab_ime.py`, `rag/collab_richtext.py`, `rag/collab_richtext_audit.py`, `rag/collab_richtext_diff.py`, `rag/collab_notify.py`, `rag/collab_notify_dispatch.py`, `rag/collab_notify_digest.py`, `rag/collab_notify_push.py`
- `rag/hybrid.py`, `rag/rerank.py`, `rag/retrieve.py`
- `rag/ingest.py`, `rag/cli.py`: ingest pipeline
- `rag/meta_store.py`: kaynak klasör/etiket sidecar (`metadata/sources.json`)
- `rag/chat_store.py`: kalıcı sohbetler
- `rag/auth.py`, `rag/oidc.py`, `rag/workspace.py`, `rag/audit.py`, `rag/metrics.py`, `rag/prometheus_sink.py`
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
- Collab: presence multi-worker (Redis) backend
- Push: VAPID OIDC / Deploy-key based secret store sync
- Judge: digest mute digest-diff annotation + actor rate limit
- Ingest: dual-write DLQ quarantine auto-requeue schedule
- Observability: inhibit equal canary Slack resolve message
