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
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

