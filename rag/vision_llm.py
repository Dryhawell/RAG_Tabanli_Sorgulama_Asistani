"""Vision-LLM: OpenAI GPT-4o ve Ollama LLaVA ile görüntü anlama."""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Literal, Optional

import requests

from app.config import (
    DEFAULT_VISION_OLLAMA_MODEL,
    DEFAULT_VISION_OPENAI_MODEL,
    OLLAMA_HOST,
    OPENAI_API_KEY,
)

VisionProvider = Literal["ollama", "openai"]


def _guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename or "")
    if mime and mime.startswith("image/"):
        return mime
    ext = os.path.splitext(filename or "")[1].lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/png")


def image_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def describe_image_openai(
    data: bytes,
    *,
    prompt: str,
    model: str = DEFAULT_VISION_OPENAI_MODEL,
    filename: str = "image.png",
    api_key: Optional[str] = None,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai paketi gerekli") from exc

    key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY ayarlı değil")

    b64 = image_to_base64(data)
    mime = _guess_mime(filename)
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()


def describe_image_ollama(
    data: bytes,
    *,
    prompt: str,
    model: str = DEFAULT_VISION_OLLAMA_MODEL,
    host: Optional[str] = None,
) -> str:
    """Ollama /api/generate images[] (llava, bakllava, moondream...)."""
    base = (host or OLLAMA_HOST).rstrip("/")
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_to_base64(data)],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    body = resp.json()
    return (body.get("response") or "").strip()


def describe_image(
    data: bytes,
    *,
    question: str = "",
    provider: VisionProvider = "openai",
    model: Optional[str] = None,
    filename: str = "image.png",
    host: Optional[str] = None,
) -> str:
    """Görüntüyü vision model ile açıklar / soruya yanıtlar."""
    q = (question or "").strip()
    prompt = (
        q
        if q
        else "Bu görüntüyü ayrıntılı açıkla. Varsa metin, tablo ve önemli nesneleri belirt."
    )
    if not q:
        prompt += " Türkçe yanıt ver."
    else:
        prompt = (
            "Görüntüye bakarak soruyu yanıtla. Metin/tablo varsa aktar.\n"
            f"Soru: {q}"
        )

    if provider == "openai":
        return describe_image_openai(
            data,
            prompt=prompt,
            model=model or DEFAULT_VISION_OPENAI_MODEL,
            filename=filename,
        )
    return describe_image_ollama(
        data,
        prompt=prompt,
        model=model or DEFAULT_VISION_OLLAMA_MODEL,
        host=host,
    )
