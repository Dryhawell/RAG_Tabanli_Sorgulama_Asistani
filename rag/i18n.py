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
        "query_rewrite": "Sorgu yeniden yazma",
        "rewrite_none": "Kapalı",
        "rewrite_hyde": "HyDE",
        "rewrite_expand": "Genişlet",
        "rewrite_both": "HyDE + Genişlet",
        "compare_panel": "Doküman karşılaştır / özetle",
        "compare_btn": "Karşılaştır",
        "summarize_btn": "Özetle",
        "compare_focus": "Karşılaştırma odağı (opsiyonel)",
        "select_sources": "Kaynaklar (en az 2)",
        "agent_tools": "Araçlar (hesap / takvim / web)",
        "source_highlight": "Kaynak vurgulama",
        "tool_traces": "Araç izleri",
        "highlighted_answer": "Kaynak vurgulu yanıt",
        "agent_memory": "Agent belleği",
        "agent_planner": "Planlı agent",
        "agent_plan": "Plan",
        "upload_image_q": "Soru için görüntü (OCR)",
        "vision_llm": "Vision-LLM (GPT-4o / LLaVA)",
        "profile_memory": "Uzun vadeli bellek",
        "profile_hits": "Profil belleği eşleşmeleri",
        "share_link": "Paylaşım linki oluştur",
        "share_link_revoke": "Linki iptal et",
        "share_link_caption": "Salt okunur paylaşım linki (giriş gerektirmez)",
        "share_view_title": "Paylaşılan sohbet (salt okunur)",
        "share_invalid": "Paylaşım linki geçersiz veya süresi dolmuş.",
        "collab_note": "İşbirlikçi not",
        "collab_save": "Notu kaydet",
        "collab_updated": "Son güncelleme: {user} — {ts}",
        "collab_conflict": "Not başka biri tarafından güncellendi; yenileyin.",
        "collab_ws_url": "Canlı WebSocket: {url}",
        "collab_remote_update": "Uzaktan güncelleme algılandı — Yenile ile çekin.",
        "collab_refresh": "Notu yenile",
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
        "query_rewrite": "Query rewrite",
        "rewrite_none": "Off",
        "rewrite_hyde": "HyDE",
        "rewrite_expand": "Expand",
        "rewrite_both": "HyDE + Expand",
        "compare_panel": "Compare / summarize documents",
        "compare_btn": "Compare",
        "summarize_btn": "Summarize",
        "compare_focus": "Comparison focus (optional)",
        "select_sources": "Sources (at least 2)",
        "agent_tools": "Tools (calc / calendar / web)",
        "source_highlight": "Source highlighting",
        "tool_traces": "Tool traces",
        "highlighted_answer": "Answer with source highlights",
        "agent_memory": "Agent memory",
        "agent_planner": "Planned agent",
        "agent_plan": "Plan",
        "upload_image_q": "Image for question (OCR)",
        "vision_llm": "Vision-LLM (GPT-4o / LLaVA)",
        "profile_memory": "Long-term memory",
        "profile_hits": "Profile memory hits",
        "share_link": "Create share link",
        "share_link_revoke": "Revoke link",
        "share_link_caption": "Read-only share link (no login required)",
        "share_view_title": "Shared chat (read-only)",
        "share_invalid": "Share link is invalid or expired.",
        "collab_note": "Collaborative note",
        "collab_save": "Save note",
        "collab_updated": "Last update: {user} — {ts}",
        "collab_conflict": "Note was updated by someone else; refresh first.",
        "collab_ws_url": "Live WebSocket: {url}",
        "collab_remote_update": "Remote update detected — pull with Refresh.",
        "collab_refresh": "Refresh note",
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
