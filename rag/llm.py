import os
from typing import Literal

LLMProvider = Literal["ollama", "openai"]


def call_ollama(model: str, prompt: str, host: str = "http://localhost:11434") -> str:
    import requests
    url = f"{host}/api/generate"
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def call_openai(model: str, prompt: str) -> str:
    try:
        import openai
    except Exception:
        raise RuntimeError("openai paketi kurulu değil. Lütfen 'pip install openai' yapın.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ayarlı değil.")
    openai.api_key = api_key
    # Basit ChatCompletion
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "Belirtilen promptu uygulayan asistan."}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp["choices"][0]["message"]["content"]


def generate_answer(provider: LLMProvider, model_name: str, prompt: str) -> str:
    if provider == "ollama":
        return call_ollama(model=model_name, prompt=prompt)
    else:
        return call_openai(model=model_name, prompt=prompt)
