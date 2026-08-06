import numpy as np
from unittest.mock import MagicMock, patch

from rag.hash_embed import HashEmbedder
from rag.profile_memory import (
    add_memory,
    ingest_session_facts,
    load_profile_memory,
    memories_context_block,
    search_memories,
)
from rag.vision import image_query_context, merge_image_into_question
from rag.vision_llm import image_to_base64


def test_image_to_base64():
    assert image_to_base64(b"abc") == "YWJj"


def test_describe_image_openai_stub():
    from rag.vision_llm import describe_image_openai

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Tabloda 3 satır var"))
    ]

    with patch("rag.vision_llm.OPENAI_API_KEY", "sk-test"), patch(
        "openai.OpenAI", return_value=fake_client
    ):
        out = describe_image_openai(
            b"\x89PNG",
            prompt="ne görüyorsun?",
            model="gpt-4o-mini",
            filename="x.png",
            api_key="sk-test",
        )
    assert "3 satır" in out
    assert fake_client.chat.completions.create.called


def test_describe_image_ollama_stub():
    from rag.vision_llm import describe_image_ollama

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"response": "Bir fatura görüyorum"}

    with patch("rag.vision_llm.requests.post", return_value=fake_resp) as post:
        out = describe_image_ollama(
            b"img",
            prompt="açıkla",
            model="llava",
            host="http://localhost:11434",
        )
    assert "fatura" in out.lower()
    assert post.called
    kwargs = post.call_args.kwargs
    assert "images" in kwargs["json"]


def test_image_query_context_vision_fallback(monkeypatch):
    monkeypatch.setattr(
        "rag.vision.ocr_image_bytes",
        lambda *a, **k: "OCR_METIN",
    )

    def boom(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr("rag.vision_llm.describe_image", boom)
    ctx = image_query_context(
        b"x",
        "a.png",
        question="nedir?",
        use_vision_llm=True,
        vision_provider="openai",
    )
    assert "OCR_METIN" in ctx
    assert "Vision-LLM atlandı" in ctx
    merged = merge_image_into_question("soru", ctx)
    assert "soru" in merged and "OCR_METIN" in merged


def test_profile_memory_add_search(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.profile_memory.METADATA_DIR", str(tmp_path))
    emb = HashEmbedder(dim=64)
    store = load_profile_memory("alice", base=str(tmp_path / "profiles"))
    add_memory(store, "Kullanıcı Türkçe tercih ediyor", kind="preference", embedder=emb)
    add_memory(store, "Yıllık izin 14 gün", kind="fact", embedder=emb)
    add_memory(store, "Kullanıcı Türkçe tercih ediyor", kind="preference", embedder=emb)  # dedupe
    assert store.size == 2

    # HashEmbedder token örtüşmesine dayanır; sorguda bellek kelimeleri olmalı
    hits = search_memories(store, "Türkçe tercih ediyor mu?", emb, top_k=2, min_score=0.01)
    assert hits
    assert "Türkçe" in hits[0].text
    block = memories_context_block(hits)
    assert "Uzun vadeli" in block

    n = ingest_session_facts(store, ["Unutma: koyu tema kullan", "x"], embedder=emb)
    assert n >= 1
    reloaded = load_profile_memory("alice", base=str(tmp_path / "profiles"))
    assert reloaded.size >= 3
