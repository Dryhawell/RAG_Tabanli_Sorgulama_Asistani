from rag.embed import (
    EMBEDDING_PRESETS,
    default_embedding_preset,
    preset_for_model,
    resolve_embedding_model,
)
from app.config import DEFAULT_EMBEDDING_MODEL, MULTILINGUAL_EMBEDDING_MODEL


def test_resolve_embedding_presets():
    assert resolve_embedding_model("mini-en") == DEFAULT_EMBEDDING_MODEL
    assert resolve_embedding_model("mini-multi") == MULTILINGUAL_EMBEDDING_MODEL
    assert resolve_embedding_model("custom/model") == "custom/model"


def test_preset_for_model():
    assert preset_for_model(DEFAULT_EMBEDDING_MODEL) == "mini-en"
    assert preset_for_model(MULTILINGUAL_EMBEDDING_MODEL) == "mini-multi"
    assert "mini-multi" in EMBEDDING_PRESETS


def test_domain_embedding_preset(monkeypatch):
    monkeypatch.setenv("RAG_DOMAIN_EMBEDDING_MODEL", "my-org/domain-model")
    monkeypatch.setenv("RAG_ENABLE_DOMAIN_EMBEDDING", "1")
    # config modülü yeniden yükle
    import importlib
    import app.config as cfg
    import rag.embed as emb

    importlib.reload(cfg)
    importlib.reload(emb)
    assert "domain" in emb.EMBEDDING_PRESETS
    assert emb.default_embedding_preset() == "domain"
    assert emb.resolve_embedding_model("domain") == "my-org/domain-model"
