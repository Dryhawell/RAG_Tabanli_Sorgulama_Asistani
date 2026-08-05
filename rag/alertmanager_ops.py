"""Alertmanager config render / reload / Slack webhook rotate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "grafana" / "alertmanager.yml.template"
DEFAULT_OUTPUT = ROOT / "grafana" / "alertmanager.rendered.yml"
DEFAULT_ENV_PATH = ROOT / "metadata" / "alertmanager.slack.env"
DEFAULT_RELOAD_URL = "http://127.0.0.1:9093/-/reload"
RENDER_SCRIPT = ROOT / "scripts" / "render_alertmanager_config.sh"


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
