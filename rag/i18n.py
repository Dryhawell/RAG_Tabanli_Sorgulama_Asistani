"""Basit UI lokalizasyonu (TR / EN)."""

from __future__ import annotations

from typing import Dict, Optional

SUPPORTED_LANGS = ("tr", "en")

_STRINGS: Dict[str, Dict[str, str]] = {
    "tr": {
        "app_title": "LLM Destekli PDF / Not Sorgulama Asistanı (RAG)",
        "settings": "Ayarlar",
        "language": "Dil",
        "login": "Giriş",
        "login_btn": "Giriş yap",
        "logout": "Çıkış yap",
        "username": "Kullanıcı adı",
        "password": "Parola",
        "sso_login": "SSO ile giriş yap",
        "documents": "Dokümanlar",
        "chats": "Sohbetler",
        "new_chat": "Yeni",
        "clear_chat": "Temizle",
        "delete": "Sil",
        "rebuild_index": "İndeksi Yeniden Oluştur",
        "upload_files": "PDF veya TXT dosyaları yükleyin",
        "ask_placeholder": "Sorunuzu yazın...",
        "index_empty": "İndeks boş. Soru sorabilmek için önce dosya yükleyin veya indeksi yeniden oluşturun.",
        "no_answer": "Bu bilgi dokümanda bulunmamaktadır",
        "sources": "Kaynaklar",
        "retrieved_sources": "Alınan kaynaklar / chunk’lar",
        "export_json": "Sohbeti JSON indir",
        "export_md": "Sohbeti Markdown indir",
        "account": "Hesap",
        "admin": "Admin",
        "hybrid_search": "Hybrid arama (BM25 + vektör)",
        "reranker": "Reranker (cross-encoder)",
        "stream_answer": "Yanıtı stream et",
        "vector_backend": "Vektör deposu",
        "invalid_credentials": "Geçersiz kullanıcı adı veya parola.",
    },
    "en": {
        "app_title": "LLM-Powered PDF / Notes Query Assistant (RAG)",
        "settings": "Settings",
        "language": "Language",
        "login": "Sign in",
        "login_btn": "Sign in",
        "logout": "Sign out",
        "username": "Username",
        "password": "Password",
        "sso_login": "Sign in with SSO",
        "documents": "Documents",
        "chats": "Chats",
        "new_chat": "New",
        "clear_chat": "Clear",
        "delete": "Delete",
        "rebuild_index": "Rebuild index",
        "upload_files": "Upload PDF or TXT files",
        "ask_placeholder": "Ask a question...",
        "index_empty": "Index is empty. Upload files or rebuild the index first.",
        "no_answer": "This information is not found in the documents",
        "sources": "Sources",
        "retrieved_sources": "Retrieved sources / chunks",
        "export_json": "Download chat as JSON",
        "export_md": "Download chat as Markdown",
        "account": "Account",
        "admin": "Admin",
        "hybrid_search": "Hybrid search (BM25 + vector)",
        "reranker": "Reranker (cross-encoder)",
        "stream_answer": "Stream answer",
        "vector_backend": "Vector store",
        "invalid_credentials": "Invalid username or password.",
    },
}

_current_lang = "tr"


def normalize_lang(lang: Optional[str]) -> str:
    code = (lang or "tr").strip().lower()
    if code.startswith("en"):
        return "en"
    return "tr"


def set_language(lang: str) -> str:
    global _current_lang
    _current_lang = normalize_lang(lang)
    return _current_lang


def get_language() -> str:
    return _current_lang


def t(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """Çeviri anahtarı. Eksikse TR, o da yoksa anahtar adı."""
    code = normalize_lang(lang or _current_lang)
    text = _STRINGS.get(code, {}).get(key)
    if text is None:
        text = _STRINGS["tr"].get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
