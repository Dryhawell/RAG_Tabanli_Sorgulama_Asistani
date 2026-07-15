from unittest.mock import MagicMock, patch

import pytest


def test_generate_answer_openai_uses_new_client():
    from rag.llm import generate_answer

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="Merhaba RAG"))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    with patch("rag.llm.OPENAI_API_KEY", "test-key"), patch(
        "openai.OpenAI", return_value=fake_client
    ):
        answer = generate_answer("openai", "gpt-4o-mini", "prompt")

    assert answer == "Merhaba RAG"
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0


def test_generate_answer_ollama():
    from rag.llm import generate_answer

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"response": "Ollama yanıtı"}

    with patch("requests.post", return_value=fake_resp) as mocked_post:
        answer = generate_answer("ollama", "phi3:mini", "prompt")

    assert answer == "Ollama yanıtı"
    assert mocked_post.called
