import os

# Kalıcılık yolları
INDEX_PATH = os.getenv("RAG_INDEX_PATH", "indexes/faiss.index")
DOCSTORE_PATH = os.getenv("RAG_DOCSTORE_PATH", "metadata/docstore.json")
DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
METADATA_DIR = os.getenv("RAG_METADATA_DIR", "metadata")
INDEXES_DIR = os.getenv("RAG_INDEXES_DIR", "indexes")

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

# Desteklenen dosya uzantıları
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
