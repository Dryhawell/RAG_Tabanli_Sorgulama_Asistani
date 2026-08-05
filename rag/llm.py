import os
from typing import Generator, Literal, Optional

from app.config import OLLAMA_HOST, OPENAI_API_KEY

LLMProvider = Literal["ollama", "openai"]


def call_ollama(
    model: str,
    prompt: str,
    host: Optional[str] = None,
    stream: bool = False,
) -> str:
    import requests

    base = host or OLLAMA_HOST
    url = f"{base.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": stream},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def stream_ollama(
    model: str,
    prompt: str,
    host: Optional[str] = None,
) -> Generator[str, None, None]:
    import json
    import requests

    base = host or OLLAMA_HOST
    url = f"{base.rstrip('/')}/api/generate"
    with requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = payload.get("response", "")
            if chunk:
                yield chunk
            if payload.get("done"):
                break


def call_openai(model: str, prompt: str, stream: bool = False) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "openai paketi kurulu değil veya uyumsuz. Lütfen 'pip install -U openai' yapın."
        ) from exc

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ayarlı değil.")

    client = OpenAI(api_key=api_key)
    if stream:
        # Non-streaming convenience path still returns full text.
        parts = []
        with client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Belirtilen promptu uygulayan asistan."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=True,
        ) as events:
            for event in events:
                delta = event.choices[0].delta.content or ""
                if delta:
                    parts.append(delta)
        return "".join(parts)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Belirtilen promptu uygulayan asistan."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def stream_openai(model: str, prompt: str) -> Generator[str, None, None]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "openai paketi kurulu değil veya uyumsuz. Lütfen 'pip install -U openai' yapın."
        ) from exc

    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ayarlı değil.")

    client = OpenAI(api_key=api_key)
    with client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Belirtilen promptu uygulayan asistan."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        stream=True,
    ) as events:
        for event in events:
            delta = event.choices[0].delta.content or ""
            if delta:
                yield delta


def generate_answer(provider: LLMProvider, model_name: str, prompt: str) -> str:
    from rag.otel import set_span_attrs, start_span

    with start_span(
        "rag.llm.generate",
        attributes={
            "rag.llm.provider": provider,
            "rag.llm.model": model_name,
            "rag.llm.stream": False,
            "rag.llm.prompt_chars": len(prompt or ""),
        },
    ) as span:
        try:
            if provider == "ollama":
                out = call_ollama(model=model_name, prompt=prompt)
            else:
                out = call_openai(model=model_name, prompt=prompt)
            set_span_attrs(span, {"rag.llm.ok": True, "rag.llm.out_chars": len(out or "")})
            return out
        except Exception as exc:
            set_span_attrs(span, {"rag.llm.ok": False, "rag.llm.error": type(exc).__name__})
            raise


def stream_answer(
    provider: LLMProvider, model_name: str, prompt: str
) -> Generator[str, None, None]:
    from rag.otel import set_span_attrs, start_span

    with start_span(
        "rag.llm.stream",
        attributes={
            "rag.llm.provider": provider,
            "rag.llm.model": model_name,
            "rag.llm.stream": True,
            "rag.llm.prompt_chars": len(prompt or ""),
        },
    ) as span:
        n = 0
        try:
            if provider == "ollama":
                gen = stream_ollama(model=model_name, prompt=prompt)
            else:
                gen = stream_openai(model=model_name, prompt=prompt)
            for chunk in gen:
                n += len(chunk or "")
                yield chunk
            set_span_attrs(span, {"rag.llm.ok": True, "rag.llm.out_chars": n})
        except Exception as exc:
            set_span_attrs(span, {"rag.llm.ok": False, "rag.llm.error": type(exc).__name__})
            raise
