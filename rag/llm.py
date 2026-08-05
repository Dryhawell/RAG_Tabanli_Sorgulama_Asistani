import os
from typing import Any, Dict, Generator, Literal, Optional, Tuple

from app.config import OLLAMA_HOST, OPENAI_API_KEY

LLMProvider = Literal["ollama", "openai"]


def _empty_usage() -> Dict[str, Any]:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def _usage_from_openai(resp: Any) -> Dict[str, Any]:
    usage = _empty_usage()
    raw = getattr(resp, "usage", None)
    if raw is None:
        return usage
    try:
        prompt = getattr(raw, "prompt_tokens", None)
        completion = getattr(raw, "completion_tokens", None)
        total = getattr(raw, "total_tokens", None)
        if prompt is None and hasattr(raw, "model_dump"):
            d = raw.model_dump()
            prompt = d.get("prompt_tokens")
            completion = d.get("completion_tokens")
            total = d.get("total_tokens")
        usage["prompt_tokens"] = int(prompt) if prompt is not None else None
        usage["completion_tokens"] = int(completion) if completion is not None else None
        if total is not None:
            usage["total_tokens"] = int(total)
        elif usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    except Exception:
        pass
    return usage


def _usage_from_ollama(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = _empty_usage()
    try:
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        if prompt is not None:
            usage["prompt_tokens"] = int(prompt)
        if completion is not None:
            usage["completion_tokens"] = int(completion)
        if usage["prompt_tokens"] is not None or usage["completion_tokens"] is not None:
            usage["total_tokens"] = int(usage["prompt_tokens"] or 0) + int(
                usage["completion_tokens"] or 0
            )
    except Exception:
        pass
    return usage


def call_ollama(
    model: str,
    prompt: str,
    host: Optional[str] = None,
    stream: bool = False,
) -> str:
    text, _ = call_ollama_with_usage(model=model, prompt=prompt, host=host, stream=stream)
    return text


def call_ollama_with_usage(
    model: str,
    prompt: str,
    host: Optional[str] = None,
    stream: bool = False,
) -> Tuple[str, Dict[str, Any]]:
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
    return data.get("response", ""), _usage_from_ollama(data if isinstance(data, dict) else {})


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
    text, _ = call_openai_with_usage(model=model, prompt=prompt, stream=stream)
    return text


def call_openai_with_usage(
    model: str, prompt: str, stream: bool = False
) -> Tuple[str, Dict[str, Any]]:
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
        return "".join(parts), _empty_usage()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Belirtilen promptu uygulayan asistan."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or "", _usage_from_openai(resp)


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


def _record_llm_usage(
    *,
    provider: str,
    model_name: str,
    usage: Dict[str, Any],
) -> None:
    try:
        from rag.metrics import record_metric

        vals = {
            "provider": provider,
            "model": model_name,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        record_metric("llm_usage", values=vals)
    except Exception:
        pass


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
                out, usage = call_ollama_with_usage(model=model_name, prompt=prompt)
            else:
                out, usage = call_openai_with_usage(model=model_name, prompt=prompt)
            attrs: Dict[str, Any] = {
                "rag.llm.ok": True,
                "rag.llm.out_chars": len(out or ""),
            }
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if usage.get(key) is not None:
                    attrs[f"rag.llm.{key}"] = usage[key]
            set_span_attrs(span, attrs)
            _record_llm_usage(provider=provider, model_name=model_name, usage=usage)
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
