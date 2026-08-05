"""Alertmanager config render / reload / Slack webhook rotate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "grafana" / "alertmanager.yml.template"
DEFAULT_OUTPUT = ROOT / "grafana" / "alertmanager.rendered.yml"
DEFAULT_ENV_PATH = ROOT / "metadata" / "alertmanager.slack.env"
DEFAULT_RELOAD_URL = "http://127.0.0.1:9093/-/reload"
DEFAULT_API_URL = "http://127.0.0.1:9093"
RENDER_SCRIPT = ROOT / "scripts" / "render_alertmanager_config.sh"


def alertmanager_api_url(base_url: Optional[str] = None) -> str:
    return (
        (base_url or "").strip()
        or os.environ.get("ALERTMANAGER_URL", "").strip()
        or os.environ.get("RAG_ALERTMANAGER_URL", "").strip()
        or DEFAULT_API_URL
    ).rstrip("/")


def _am_request(
    method: str,
    path: str,
    *,
    body: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    root = alertmanager_api_url(base_url)
    url = f"{root}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        import json as _json

        data = _json.dumps(body).encode("utf-8")
    try:
        req = request.Request(url, data=data, method=method.upper(), headers=headers)
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = getattr(resp, "status", None) or resp.getcode()
            parsed: Any = None
            if raw:
                try:
                    import json as _json

                    parsed = _json.loads(raw.decode("utf-8"))
                except Exception:
                    parsed = raw.decode("utf-8", errors="replace")
            return {"ok": 200 <= int(code) < 300, "status": int(code), "url": url, "data": parsed}
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return {
            "ok": False,
            "status": int(exc.code),
            "url": url,
            "error": detail or str(exc),
        }
    except Exception as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__, "detail": str(exc)}


def parse_silence_matcher(raw: str) -> Dict[str, Any]:
    """name=value veya name=~regex → Alertmanager matcher."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty_matcher")
    is_regex = False
    if "=~" in text:
        name, value = text.split("=~", 1)
        is_regex = True
    elif "=" in text:
        name, value = text.split("=", 1)
    else:
        raise ValueError(f"bad_matcher:{text}")
    name = name.strip()
    value = value.strip()
    if not name:
        raise ValueError("matcher_name_missing")
    return {
        "name": name,
        "value": value,
        "isRegex": is_regex,
        "isEqual": True,
    }


def parse_duration_sec(raw: str) -> float:
    text = (raw or "").strip().lower()
    if not text:
        raise ValueError("empty_duration")
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    if text.endswith("h"):
        return float(text[:-1]) * 3600.0
    if text.endswith("d"):
        return float(text[:-1]) * 86400.0
    return float(text)


def create_silence(
    *,
    matchers: List[Dict[str, Any]],
    starts_at: Optional[str] = None,
    ends_at: Optional[str] = None,
    duration_sec: Optional[float] = None,
    created_by: str = "rag-cli",
    comment: str = "",
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """POST /api/v2/silences."""
    from datetime import datetime, timedelta, timezone

    if not matchers:
        return {"ok": False, "error": "matchers_required"}
    now = datetime.now(timezone.utc)
    start = starts_at or now.isoformat().replace("+00:00", "Z")
    if ends_at:
        end = ends_at
    else:
        secs = float(duration_sec if duration_sec is not None else 3600.0)
        end = (now + timedelta(seconds=secs)).isoformat().replace("+00:00", "Z")
    body = {
        "matchers": matchers,
        "startsAt": start,
        "endsAt": end,
        "createdBy": created_by or "rag-cli",
        "comment": comment or "rag silence",
    }
    resp = _am_request(
        "POST", "/api/v2/silences", body=body, base_url=base_url, timeout=timeout
    )
    silence_id = None
    data = resp.get("data")
    if isinstance(data, dict):
        silence_id = data.get("silenceID") or data.get("id")
    elif isinstance(data, str) and data.strip():
        silence_id = data.strip().strip('"')
    return {
        "ok": bool(resp.get("ok")),
        "silenceID": silence_id,
        "status": resp.get("status"),
        "url": resp.get("url"),
        "error": resp.get("error"),
        "request": body,
        "response": data,
    }


def list_silences(*, base_url: Optional[str] = None, timeout: float = 5.0) -> Dict[str, Any]:
    resp = _am_request("GET", "/api/v2/silences", base_url=base_url, timeout=timeout)
    data = resp.get("data")
    return {
        "ok": bool(resp.get("ok")),
        "silences": data if isinstance(data, list) else [],
        "status": resp.get("status"),
        "error": resp.get("error"),
    }


def delete_silence(
    silence_id: str,
    *,
    base_url: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    sid = (silence_id or "").strip()
    if not sid:
        return {"ok": False, "error": "silence_id_missing"}
    from urllib.parse import quote

    resp = _am_request(
        "DELETE",
        f"/api/v2/silence/{quote(sid, safe='')}",
        base_url=base_url,
        timeout=timeout,
    )
    return {
        "ok": bool(resp.get("ok")),
        "silenceID": sid,
        "status": resp.get("status"),
        "error": resp.get("error"),
    }


def render_alertmanager_config(
    *,
    template: Optional[str] = None,
    output: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Template → YAML (shell script / envsubst)."""
    env = os.environ.copy()
    if slack_webhook is not None:
        env["RAG_ALERTMANAGER_SLACK_WEBHOOK"] = slack_webhook
    if webhook_url is not None:
        env["RAG_ALERTMANAGER_WEBHOOK_URL"] = webhook_url
    tmpl = template or env.get("ALERTMANAGER_TEMPLATE") or str(DEFAULT_TEMPLATE)
    out = output or env.get("ALERTMANAGER_OUTPUT") or str(DEFAULT_OUTPUT)
    env["ALERTMANAGER_TEMPLATE"] = tmpl
    env["ALERTMANAGER_OUTPUT"] = out
    if not Path(RENDER_SCRIPT).is_file():
        return {"ok": False, "error": "render_script_missing", "script": str(RENDER_SCRIPT)}
    proc = subprocess.run(
        ["sh", str(RENDER_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": out,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def reload_alertmanager(
    *,
    url: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """POST /-/reload — çalışan Alertmanager config yeniler."""
    target = (
        (url or "").strip()
        or os.environ.get("ALERTMANAGER_RELOAD_URL", "").strip()
        or DEFAULT_RELOAD_URL
    )
    try:
        req = request.Request(target, method="POST", data=b"")
        with request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return {"ok": 200 <= int(code) < 300, "status": int(code), "url": target}
    except error.HTTPError as exc:
        return {"ok": False, "status": int(exc.code), "url": target, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "url": target, "error": type(exc).__name__, "detail": str(exc)}


def write_slack_webhook_env(
    webhook: str,
    *,
    path: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> str:
    """Slack webhook (ve opsiyonel genel webhook) env dosyası yazar."""
    out = Path(path or os.environ.get("RAG_ALERTMANAGER_ENV_PATH") or DEFAULT_ENV_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Alertmanager Slack webhook — generated",
        f"RAG_ALERTMANAGER_SLACK_WEBHOOK={webhook.strip()}",
    ]
    wh = (webhook_url if webhook_url is not None else os.environ.get("RAG_ALERTMANAGER_WEBHOOK_URL", "")).strip()
    if wh:
        lines.append(f"RAG_ALERTMANAGER_WEBHOOK_URL={wh}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def rotate_alertmanager_slack_webhook(
    webhook: str,
    *,
    write_env: Optional[str] = None,
    template: Optional[str] = None,
    output: Optional[str] = None,
    reload: bool = True,
    reload_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Webhook kaydet → config render → (opsiyonel) Alertmanager reload."""
    wh = (webhook or "").strip()
    if not wh:
        return {"ok": False, "error": "webhook_missing"}
    env_path = write_slack_webhook_env(wh, path=write_env, webhook_url=webhook_url)
    rendered = render_alertmanager_config(
        template=template,
        output=output,
        slack_webhook=wh,
        webhook_url=webhook_url,
    )
    result: Dict[str, Any] = {
        "ok": bool(rendered.get("ok")),
        "env_path": env_path,
        "render": rendered,
    }
    if reload and rendered.get("ok"):
        reloaded = reload_alertmanager(url=reload_url)
        result["reload"] = reloaded
        result["ok"] = bool(reloaded.get("ok"))
    elif not reload:
        result["reload"] = {"ok": True, "skipped": True}
    return result
