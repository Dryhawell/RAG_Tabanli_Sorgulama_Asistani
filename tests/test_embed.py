from rag.embed import EMBEDDING_PRESETS, preset_for_model, resolve_embedding_model
from app.config import DEFAULT_EMBEDDING_MODEL, MULTILINGUAL_EMBEDDING_MODEL


def test_resolve_embedding_presets():
    assert resolve_embedding_model("mini-en") == DEFAULT_EMBEDDING_MODEL
    assert resolve_embedding_model("mini-multi") == MULTILINGUAL_EMBEDDING_MODEL
    assert resolve_embedding_model("custom/model") == "custom/model"


def test_preset_for_model():
    assert preset_for_model(DEFAULT_EMBEDDING_MODEL) == "mini-en"
    assert preset_for_model(MULTILINGUAL_EMBEDDING_MODEL) == "mini-multi"
    assert "mini-multi" in EMBEDDING_PRESETS
