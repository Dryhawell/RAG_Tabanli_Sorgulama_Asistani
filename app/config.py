import os

# Kalıcılık yolları
INDEX_PATH = os.getenv("RAG_INDEX_PATH", "indexes/faiss.index")
DOCSTORE_PATH = os.getenv("RAG_DOCSTORE_PATH", "metadata/docstore.json")
DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
METADATA_DIR = os.getenv("RAG_METADATA_DIR", "metadata")
INDEXES_DIR = os.getenv("RAG_INDEXES_DIR", "indexes")
CHAT_DIR = os.getenv("RAG_CHAT_DIR", os.path.join(METADATA_DIR, "chats"))
USERS_PATH = os.getenv("RAG_USERS_PATH", os.path.join(METADATA_DIR, "users.json"))

# Auth (çok kullanıcılı)
ENABLE_AUTH = os.getenv("RAG_ENABLE_AUTH", "0") not in {"0", "false", "False"}
# true: tüm kullanıcılar aynı data/indeksi paylaşır
# false: her kullanıcının kendi data/indexes/metadata alanı olur
AUTH_SHARED_INDEX = os.getenv("RAG_AUTH_SHARED_INDEX", "1") not in {"0", "false", "False"}
# user rolünün dosya yükleme/silme yetkisi (yalnızca paylaşımlı indekste anlamlı)
AUTH_USER_CAN_INGEST = os.getenv("RAG_AUTH_USER_CAN_INGEST", "0") not in {"0", "false", "False"}
# İlk kurulum için: "admin:parola" — users.json yoksa oluşturulur
AUTH_BOOTSTRAP_ADMIN = os.getenv("RAG_AUTH_BOOTSTRAP_ADMIN", "admin:admin")
# Çok kiracılı (tenant) izolasyon
ENABLE_TENANTS = os.getenv("RAG_ENABLE_TENANTS", "0") not in {"0", "false", "False"}
DEFAULT_TENANT = os.getenv("RAG_DEFAULT_TENANT", "default")
# Audit log
ENABLE_AUDIT = os.getenv("RAG_ENABLE_AUDIT", "1") not in {"0", "false", "False"}
AUDIT_LOG_PATH = os.getenv("RAG_AUDIT_LOG_PATH", os.path.join(METADATA_DIR, "audit.jsonl"))
# Metrikler / observability
ENABLE_METRICS = os.getenv("RAG_ENABLE_METRICS", "1") not in {"0", "false", "False"}
METRICS_PATH = os.getenv("RAG_METRICS_PATH", os.path.join(METADATA_DIR, "metrics.jsonl"))

# OIDC / SSO (Authorization Code)
ENABLE_OIDC = os.getenv("RAG_ENABLE_OIDC", "0") not in {"0", "false", "False"}
OIDC_ENABLED = ENABLE_OIDC  # alias
OIDC_ONLY = os.getenv("RAG_OIDC_ONLY", "0") not in {"0", "false", "False"}
OIDC_ISSUER = os.getenv("RAG_OIDC_ISSUER", "")
OIDC_CLIENT_ID = os.getenv("RAG_OIDC_CLIENT_ID", "")
OIDC_CLIENT_SECRET = os.getenv("RAG_OIDC_CLIENT_SECRET", "")
OIDC_REDIRECT_URI = os.getenv(
    "RAG_OIDC_REDIRECT_URI",
    "http://localhost:8501",
)
OIDC_SCOPES = os.getenv("RAG_OIDC_SCOPES", "openid profile email")
OIDC_USERNAME_CLAIM = os.getenv("RAG_OIDC_USERNAME_CLAIM", "preferred_username")
OIDC_TENANT_CLAIM = os.getenv("RAG_OIDC_TENANT_CLAIM", "tenant_id")
OIDC_ADMIN_GROUPS = os.getenv("RAG_OIDC_ADMIN_GROUPS", "rag-admins,admin")
OIDC_DEFAULT_ROLE = os.getenv("RAG_OIDC_DEFAULT_ROLE", "user")
OIDC_AUTO_PROVISION = os.getenv("RAG_OIDC_AUTO_PROVISION", "1") not in {
    "0",
    "false",
    "False",
}
# id_token JWKS imza doğrulama (1 = zorunlu)
OIDC_VERIFY_JWKS = os.getenv("RAG_OIDC_VERIFY_JWKS", "1") not in {"0", "false", "False"}

# Prometheus metrik sink
ENABLE_PROMETHEUS = os.getenv("RAG_ENABLE_PROMETHEUS", "0") not in {"0", "false", "False"}
PROMETHEUS_PORT = int(os.getenv("RAG_PROMETHEUS_PORT", "9108"))
PROMETHEUS_ADDR = os.getenv("RAG_PROMETHEUS_ADDR", "0.0.0.0")

# Vektör deposu: faiss (varsayılan) | qdrant
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "faiss").strip().lower()
QDRANT_URL = os.getenv("RAG_QDRANT_URL", "")  # örn. http://localhost:6333
QDRANT_PATH = os.getenv("RAG_QDRANT_PATH", os.path.join(INDEXES_DIR, "qdrant_local"))
QDRANT_API_KEY = os.getenv("RAG_QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks")

# UI dili: tr | en
DEFAULT_UI_LANG = os.getenv("RAG_UI_LANG", "tr").strip().lower()

# Embedding
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
MULTILINGUAL_EMBEDDING_MODEL = os.getenv(
    "RAG_MULTILINGUAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Retrieval
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
NO_ANSWER_THRESHOLD = float(os.getenv("RAG_NO_ANSWER_THRESHOLD", "0.30"))
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.65"))  # vektör ağırlığı
DEFAULT_RERANKER_MODEL = os.getenv(
    "RAG_RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
ENABLE_RERANKER = os.getenv("RAG_ENABLE_RERANKER", "1") not in {"0", "false", "False"}

# Chunking
CHUNK_SIZE_WORDS = int(os.getenv("RAG_CHUNK_SIZE_WORDS", "700"))
CHUNK_OVERLAP_RATIO = float(os.getenv("RAG_CHUNK_OVERLAP_RATIO", "0.12"))
MIN_CHUNK_WORDS = int(os.getenv("RAG_MIN_CHUNK_WORDS", "200"))

# LLM
DEFAULT_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama")
DEFAULT_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "phi3:mini")
DEFAULT_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OCR
ENABLE_OCR = os.getenv("RAG_ENABLE_OCR", "1") not in {"0", "false", "False"}
OCR_LANGS = os.getenv("RAG_OCR_LANGS", "tur+eng")
OCR_MIN_CHARS = int(os.getenv("RAG_OCR_MIN_CHARS", "40"))  # sayfa metni bundan kısaysa OCR dene

# Layout-aware PDF (blok okuma sırası + tablo → markdown)
ENABLE_LAYOUT_PDF = os.getenv("RAG_ENABLE_LAYOUT_PDF", "1") not in {"0", "false", "False"}
LAYOUT_TABLE_MIN_ROWS = int(os.getenv("RAG_LAYOUT_TABLE_MIN_ROWS", "2"))

# Desteklenen dosya uzantıları
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# Multimodal
ENABLE_IMAGE_OCR = os.getenv("RAG_ENABLE_IMAGE_OCR", "1") not in {"0", "false", "False"}
ENABLE_TABLE_BOOST = os.getenv("RAG_ENABLE_TABLE_BOOST", "1") not in {"0", "false", "False"}

# Sorgu yeniden yazma / HyDE
ENABLE_QUERY_REWRITE = os.getenv("RAG_ENABLE_QUERY_REWRITE", "0") not in {
    "0",
    "false",
    "False",
}
# none | hyde | expand | hyde+expand
DEFAULT_QUERY_REWRITE_MODE = os.getenv("RAG_QUERY_REWRITE_MODE", "hyde")

# Agentic araçlar + kaynak vurgulama
ENABLE_AGENT_TOOLS = os.getenv("RAG_ENABLE_AGENT_TOOLS", "0") not in {
    "0",
    "false",
    "False",
}
AGENT_MAX_STEPS = int(os.getenv("RAG_AGENT_MAX_STEPS", "3"))
ENABLE_SOURCE_HIGHLIGHT = os.getenv("RAG_ENABLE_SOURCE_HIGHLIGHT", "1") not in {
    "0",
    "false",
    "False",
}
HIGHLIGHT_MIN_TOKENS = int(os.getenv("RAG_HIGHLIGHT_MIN_TOKENS", "4"))

# Agent bellek / planlama
ENABLE_AGENT_MEMORY = os.getenv("RAG_ENABLE_AGENT_MEMORY", "1") not in {
    "0",
    "false",
    "False",
}
ENABLE_AGENT_PLANNER = os.getenv("RAG_ENABLE_AGENT_PLANNER", "0") not in {
    "0",
    "false",
    "False",
}

# Vision-LLM (GPT-4o / LLaVA)
ENABLE_VISION_LLM = os.getenv("RAG_ENABLE_VISION_LLM", "0") not in {"0", "false", "False"}
DEFAULT_VISION_OPENAI_MODEL = os.getenv("RAG_VISION_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_VISION_OLLAMA_MODEL = os.getenv("RAG_VISION_OLLAMA_MODEL", "llava")

# Uzun vadeli vektör bellek (kullanıcı profili)
ENABLE_PROFILE_MEMORY = os.getenv("RAG_ENABLE_PROFILE_MEMORY", "1") not in {
    "0",
    "false",
    "False",
}
PROFILE_MEMORY_TOP_K = int(os.getenv("RAG_PROFILE_MEMORY_TOP_K", "4"))
PROFILE_MEMORY_MIN_SCORE = float(os.getenv("RAG_PROFILE_MEMORY_MIN_SCORE", "0.28"))

# Paylaşım linkleri + işbirlikçi not
ENABLE_SHARE_LINKS = os.getenv("RAG_ENABLE_SHARE_LINKS", "1") not in {
    "0",
    "false",
    "False",
}
SHARE_LINK_DEFAULT_TTL_DAYS = int(os.getenv("RAG_SHARE_LINK_TTL_DAYS", "7"))
PUBLIC_BASE_URL = os.getenv("RAG_PUBLIC_BASE_URL", "http://localhost:8501")
ENABLE_COLLAB_NOTES = os.getenv("RAG_ENABLE_COLLAB_NOTES", "1") not in {
    "0",
    "false",
    "False",
}
ENABLE_COLLAB_WS = os.getenv("RAG_ENABLE_COLLAB_WS", "0") not in {
    "0",
    "false",
    "False",
}
COLLAB_WS_HOST = os.getenv("RAG_COLLAB_WS_HOST", "0.0.0.0")
COLLAB_WS_PORT = int(os.getenv("RAG_COLLAB_WS_PORT", "8765"))
COLLAB_WS_PUBLIC_HOST = os.getenv("RAG_COLLAB_WS_PUBLIC_HOST", "localhost")
# Web Push aynı-origin SW / register HTTP (WS portundan ayrı)
COLLAB_HTTP_PORT = int(os.getenv("RAG_COLLAB_HTTP_PORT", "8766"))
ENABLE_COLLAB_HTTP = os.getenv("RAG_ENABLE_COLLAB_HTTP", "1") not in {
    "0",
    "false",
    "False",
}
ENABLE_COLLAB_CRDT = os.getenv("RAG_ENABLE_COLLAB_CRDT", "1") not in {
    "0",
    "false",
    "False",
}
EMBED_FINETUNE_OUTPUT_DIR = os.getenv(
    "RAG_EMBED_FINETUNE_OUTPUT_DIR",
    os.path.join("models", "embed-finetuned"),
)
EMBED_PIPELINE_REPORT_PATH = os.getenv(
    "RAG_EMBED_PIPELINE_REPORT_PATH",
    os.path.join(METADATA_DIR, "embed_pipeline_report.json"),
)

# Domain / fine-tuned embedding (yerel veya Hub model yolu)
DOMAIN_EMBEDDING_MODEL = os.getenv("RAG_DOMAIN_EMBEDDING_MODEL", "").strip()
ENABLE_DOMAIN_EMBEDDING = os.getenv("RAG_ENABLE_DOMAIN_EMBEDDING", "0") not in {
    "0",
    "false",
    "False",
}

# Otomatik domain veri toplama
ENABLE_DOMAIN_COLLECT = os.getenv("RAG_ENABLE_DOMAIN_COLLECT", "1") not in {
    "0",
    "false",
    "False",
}
DOMAIN_PAIRS_PATH = os.getenv(
    "RAG_DOMAIN_PAIRS_PATH",
    os.path.join(METADATA_DIR, "domain_training", "pairs.jsonl"),
)
DOMAIN_COLLECT_MIN_GATE = float(os.getenv("RAG_DOMAIN_COLLECT_MIN_GATE", "0.35"))
ENABLE_COLLAB_LIVE_EDITOR = os.getenv("RAG_ENABLE_COLLAB_LIVE_EDITOR", "1") not in {
    "0",
    "false",
    "False",
}
ENABLE_COLLAB_RICHTEXT = os.getenv("RAG_ENABLE_COLLAB_RICHTEXT", "1") not in {
    "0",
    "false",
    "False",
}

# Federated embedding havuzu
ENABLE_FEDERATED_POOL = os.getenv("RAG_ENABLE_FEDERATED_POOL", "1") not in {
    "0",
    "false",
    "False",
}
FEDERATED_POOL_PATH = os.getenv(
    "RAG_FEDERATED_POOL_PATH",
    os.path.join(METADATA_DIR, "federated", "training_pool.jsonl"),
)
FEDERATED_MIN_PER_TENANT = int(os.getenv("RAG_FEDERATED_MIN_PER_TENANT", "0"))

# Cross-tenant privacy (DP-SGD lite + secure aggregation PoC)
ENABLE_FEDERATED_PRIVACY = os.getenv("RAG_ENABLE_FEDERATED_PRIVACY", "1") not in {
    "0",
    "false",
    "False",
}
FEDERATED_DP_NOISE = float(os.getenv("RAG_FEDERATED_DP_NOISE", "1.0"))
FEDERATED_DP_CLIP = float(os.getenv("RAG_FEDERATED_DP_CLIP", "1.0"))
FEDERATED_SECRET = os.getenv("RAG_FEDERATED_SECRET", "rag-federated")
PRIVATE_FEDERATED_POOL_PATH = os.getenv(
    "RAG_PRIVATE_FEDERATED_POOL_PATH",
    os.path.join(METADATA_DIR, "federated", "private_training_pool.jsonl"),
)

# Opacus / DP-SGD eğitim
ENABLE_DP_TRAIN = os.getenv("RAG_ENABLE_DP_TRAIN", "1") not in {"0", "false", "False"}
DP_TRAIN_OUTPUT_DIR = os.getenv(
    "RAG_DP_TRAIN_OUTPUT_DIR",
    os.path.join("models", "dp-embed"),
)
DP_TRAIN_NOISE = float(os.getenv("RAG_DP_TRAIN_NOISE", "1.0"))
DP_TRAIN_MAX_GRAD_NORM = float(os.getenv("RAG_DP_TRAIN_MAX_GRAD_NORM", "1.0"))
DP_TRAIN_DELTA = float(os.getenv("RAG_DP_TRAIN_DELTA", "1e-5"))
DP_TRAIN_USE_OPACUS = os.getenv("RAG_DP_TRAIN_USE_OPACUS", "1") not in {
    "0",
    "false",
    "False",
}
ST_DP_TRAIN_OUTPUT_DIR = os.getenv(
    "RAG_ST_DP_TRAIN_OUTPUT_DIR",
    os.path.join("models", "st-dp-embed"),
)
ST_DP_HEAD_DIM = int(os.getenv("RAG_ST_DP_HEAD_DIM", "64"))
ST_DP_MAX_SEQ_LENGTH = int(os.getenv("RAG_ST_DP_MAX_SEQ_LENGTH", "64"))
ST_DP_FREEZE_BACKBONE = os.getenv("RAG_ST_DP_FREEZE_BACKBONE", "1") not in {
    "0",
    "false",
    "False",
}
LORA_DP_TRAIN_OUTPUT_DIR = os.getenv(
    "RAG_LORA_DP_TRAIN_OUTPUT_DIR",
    os.path.join("models", "lora-dp-embed"),
)
LORA_DP_RANK = int(os.getenv("RAG_LORA_DP_RANK", "8"))
LORA_DP_ALPHA = int(os.getenv("RAG_LORA_DP_ALPHA", "16"))
LORA_DP_MOCK = os.getenv("RAG_LORA_DP_MOCK", "0") not in {"0", "false", "False"}
LORA_DP_OPACUS_PRODUCTION = os.getenv("RAG_LORA_DP_OPACUS_PRODUCTION", "0") not in {
    "0",
    "false",
    "False",
}
LORA_DP_SECURE_MODE = os.getenv("RAG_LORA_DP_SECURE_MODE", "0") not in {
    "0",
    "false",
    "False",
}
LORA_DP_GRAD_SAMPLE_MODE = os.getenv("RAG_LORA_DP_GRAD_SAMPLE_MODE", "hooks")
LORA_DP_EVAL_MIN_ACCURACY = float(os.getenv("RAG_LORA_DP_EVAL_MIN_ACCURACY", "0.0"))
LORA_DP_EVAL_MIN_DELTA = float(os.getenv("RAG_LORA_DP_EVAL_MIN_DELTA", "0.0"))
JUDGE_MIN_ACCURACY = float(os.getenv("RAG_JUDGE_MIN_ACCURACY", "1.0"))
JUDGE_MODE = os.getenv("RAG_JUDGE_MODE", "heuristic").strip().lower() or "heuristic"

# Collab bildirim dağıtımı (e-posta / webhook)
ENABLE_COLLAB_NOTIFY_DISPATCH = os.getenv("RAG_ENABLE_COLLAB_NOTIFY_DISPATCH", "1") not in {
    "0",
    "false",
    "False",
}
NOTIFY_SMTP_HOST = os.getenv("RAG_NOTIFY_SMTP_HOST", "").strip()
NOTIFY_SMTP_PORT = int(os.getenv("RAG_NOTIFY_SMTP_PORT", "587"))
NOTIFY_SMTP_USER = os.getenv("RAG_NOTIFY_SMTP_USER", "").strip()
NOTIFY_SMTP_PASSWORD = os.getenv("RAG_NOTIFY_SMTP_PASSWORD", "").strip()
NOTIFY_FROM_EMAIL = os.getenv("RAG_NOTIFY_FROM_EMAIL", "noreply@localhost").strip()
NOTIFY_WEBHOOK_URL = os.getenv("RAG_NOTIFY_WEBHOOK_URL", "").strip()
NOTIFY_DIGEST_HOURS = int(os.getenv("RAG_NOTIFY_DIGEST_HOURS", "24"))
NOTIFY_DIGEST_MENTIONS_ONLY = os.getenv("RAG_NOTIFY_DIGEST_MENTIONS_ONLY", "0") not in {
    "0",
    "false",
    "False",
}
NOTIFY_DIGEST_GROUP_BY = os.getenv("RAG_NOTIFY_DIGEST_GROUP_BY", "thread").strip().lower()
NOTIFY_DIGEST_MIN_PER_WORKSPACE = int(os.getenv("RAG_NOTIFY_DIGEST_MIN_PER_WORKSPACE", "1"))
NOTIFY_DIGEST_HTML = os.getenv("RAG_NOTIFY_DIGEST_HTML", "1") not in {
    "0",
    "false",
    "False",
}
# Quiet hours: HH:MM-HH:MM (UTC), örn. 22:00-07:00 — boş = kapalı
NOTIFY_DIGEST_QUIET_HOURS = os.getenv("RAG_NOTIFY_DIGEST_QUIET_HOURS", "").strip()
# Quiet hours timezone (IANA, örn. Europe/Istanbul); boş = UTC
NOTIFY_DIGEST_TIMEZONE = os.getenv("RAG_NOTIFY_DIGEST_TIMEZONE", "UTC").strip() or "UTC"
# Tenant → timezone JSON veya "id:tz,id2:tz2"
NOTIFY_TENANT_TIMEZONES = os.getenv("RAG_NOTIFY_TENANT_TIMEZONES", "").strip()
# Quiet hours sırasında Slack/Teams thread reply özeti
NOTIFY_DIGEST_THREAD_REPLY = os.getenv("RAG_NOTIFY_DIGEST_THREAD_REPLY", "1") not in {
    "0",
    "false",
    "False",
}
NOTIFY_DIGEST_SLACK_THREAD_TS = os.getenv("RAG_NOTIFY_DIGEST_SLACK_THREAD_TS", "").strip()
NOTIFY_DIGEST_TEAMS_REPLY_ID = os.getenv("RAG_NOTIFY_DIGEST_TEAMS_REPLY_ID", "").strip()
NOTIFY_SLACK_BOT_TOKEN = os.getenv("RAG_NOTIFY_SLACK_BOT_TOKEN", "").strip()
NOTIFY_SLACK_CHANNEL = os.getenv("RAG_NOTIFY_SLACK_CHANNEL", "").strip()
# Mobil push PoC (FCM HTTP / generic push endpoint)
NOTIFY_PUSH_URL = os.getenv("RAG_NOTIFY_PUSH_URL", "").strip()
NOTIFY_PUSH_API_KEY = os.getenv("RAG_NOTIFY_PUSH_API_KEY", "").strip()
NOTIFY_PUSH_PROVIDER = os.getenv("RAG_NOTIFY_PUSH_PROVIDER", "generic").strip().lower()
NOTIFY_PUSH_TOKEN_TTL_DAYS = int(os.getenv("RAG_NOTIFY_PUSH_TOKEN_TTL_DAYS", "90"))
NOTIFY_PUSH_FCM_PROJECT_ID = os.getenv("RAG_NOTIFY_PUSH_FCM_PROJECT_ID", "").strip()
NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON = os.getenv(
    "RAG_NOTIFY_PUSH_FCM_SERVICE_ACCOUNT_JSON", ""
).strip()
# APNs HTTP/2 native (.p8 JWT) — boşsa gateway URL kullanılır
NOTIFY_PUSH_APNS_KEY_ID = os.getenv("RAG_NOTIFY_PUSH_APNS_KEY_ID", "").strip()
NOTIFY_PUSH_APNS_TEAM_ID = os.getenv("RAG_NOTIFY_PUSH_APNS_TEAM_ID", "").strip()
NOTIFY_PUSH_APNS_TOPIC = os.getenv("RAG_NOTIFY_PUSH_APNS_TOPIC", "").strip()
NOTIFY_PUSH_APNS_P8_PATH = os.getenv("RAG_NOTIFY_PUSH_APNS_P8_PATH", "").strip()
NOTIFY_PUSH_APNS_P8_CONTENT = os.getenv("RAG_NOTIFY_PUSH_APNS_P8_CONTENT", "").strip()
NOTIFY_PUSH_APNS_USE_SANDBOX = os.getenv("RAG_NOTIFY_PUSH_APNS_USE_SANDBOX", "0") not in {
    "0",
    "false",
    "False",
}
# Web Push (VAPID)
NOTIFY_PUSH_VAPID_PUBLIC = os.getenv("RAG_NOTIFY_PUSH_VAPID_PUBLIC", "").strip()
NOTIFY_PUSH_VAPID_PRIVATE = os.getenv("RAG_NOTIFY_PUSH_VAPID_PRIVATE", "").strip()
NOTIFY_PUSH_VAPID_SUBJECT = os.getenv(
    "RAG_NOTIFY_PUSH_VAPID_SUBJECT",
    "mailto:admin@localhost",
).strip()
# Digest rapor webhook alert eşikleri
DIGEST_ALERT_SKIP_THRESHOLD = float(
    os.getenv("RAG_DIGEST_ALERT_SKIP_THRESHOLD", "0.5")
)
DIGEST_ALERT_FAIL_RATE = float(os.getenv("RAG_DIGEST_ALERT_FAIL_RATE", "0.2"))
DIGEST_ALERT_MIN_SAMPLES = int(os.getenv("RAG_DIGEST_ALERT_MIN_SAMPLES", "5"))
DIGEST_ALERT_WEBHOOK_URL = os.getenv("RAG_DIGEST_ALERT_WEBHOOK_URL", "").strip()
DIGEST_ALERT_WEBHOOKS_JSON = os.getenv("RAG_DIGEST_ALERT_WEBHOOKS_JSON", "").strip()
# Presence snapshot TTL (saniye)
COLLAB_PRESENCE_TTL_SEC = int(os.getenv("RAG_COLLAB_PRESENCE_TTL_SEC", "90"))
LORA_DP_EVAL_AUTO_ROLLBACK = os.getenv("RAG_LORA_DP_EVAL_AUTO_ROLLBACK", "0") not in {
    "0",
    "false",
    "False",
}

