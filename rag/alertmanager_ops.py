"""Alertmanager config render / reload / Slack webhook rotate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
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


def list_alerts(
    *,
    base_url: Optional[str] = None,
    active: bool = True,
    silenced: bool = False,
    inhibited: bool = False,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """GET /api/v2/alerts — canlı firing alert listesi."""
    from urllib.parse import urlencode

    qs = urlencode(
        {
            "active": str(bool(active)).lower(),
            "silenced": str(bool(silenced)).lower(),
            "inhibited": str(bool(inhibited)).lower(),
        }
    )
    resp = _am_request(
        "GET", f"/api/v2/alerts?{qs}", base_url=base_url, timeout=timeout
    )
    data = resp.get("data")
    return {
        "ok": bool(resp.get("ok")),
        "alerts": data if isinstance(data, list) else [],
        "count": len(data) if isinstance(data, list) else 0,
        "status": resp.get("status"),
        "error": resp.get("error"),
        "url": resp.get("url"),
    }


def extract_live_alert_labels(
    alerts: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Alertmanager /api/v2/alerts satırlarından label setleri çıkar."""
    found: List[Dict[str, Any]] = []
    for item in alerts or []:
        if not isinstance(item, dict):
            continue
        labels = item.get("labels")
        if isinstance(labels, dict) and labels:
            found.append({"labels": dict(labels), "source": "alertmanager"})
    return found


def suggest_equal_labels(
    label_sets: Sequence[Dict[str, Any]],
    *,
    prefer: Sequence[str] = ("alertname", "service", "secondary_backend", "tenant"),
    min_count: int = 2,
    max_labels: int = 4,
) -> List[str]:
    """Canlı/statik label setlerinden inhibit `equal` öner."""
    blocklist = {
        "annotations",
        "summary",
        "description",
        "title",
        "message",
        "dashboard",
        "runbook_url",
        "value",
    }
    counts: Dict[str, int] = {}
    for item in label_sets:
        if not isinstance(item, dict):
            continue
        # Top-level alertname (Grafana extract)
        top_name = str(item.get("alertname") or "").strip()
        if top_name:
            counts["alertname"] = counts.get("alertname", 0) + 1
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        if not isinstance(labels, dict):
            continue
        for key in labels.keys():
            k = str(key).strip()
            if not k or k == "severity" or k in blocklist:
                continue
            counts[k] = counts.get(k, 0) + 1

    chosen: List[str] = []
    for key in prefer:
        if counts.get(key, 0) >= min_count and key not in chosen:
            chosen.append(key)
    extras = sorted(
        ((k, c) for k, c in counts.items() if k not in chosen and c >= min_count),
        key=lambda kv: (-kv[1], kv[0]),
    )
    for k, _c in extras:
        if len(chosen) >= max_labels:
            break
        chosen.append(k)
    if not chosen:
        for key in ("alertname", "service"):
            if key in counts and key not in chosen:
                chosen.append(key)
        if not chosen:
            chosen = ["alertname", "service"]
    return chosen[:max_labels]


def tune_inhibit_equal_from_live(
    *,
    base_url: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
    prefer: Optional[Sequence[str]] = None,
    fallback_equal: Sequence[str] = ("alertname", "service"),
    min_count: int = 1,
    include_static: bool = True,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Canlı Alertmanager alert'lerinden equal label öner (+ opsiyonel Grafana static)."""
    live = list_alerts(base_url=base_url, timeout=timeout)
    label_sets: List[Dict[str, Any]] = []
    live_count = 0
    if live.get("ok"):
        live_sets = extract_live_alert_labels(live.get("alerts") or [])
        live_count = len(live_sets)
        label_sets.extend(live_sets)
    static_count = 0
    if include_static:
        static_sets = extract_grafana_alert_labels(paths)
        static_count = len(static_sets)
        label_sets.extend(static_sets)

    if not label_sets:
        equal = list(fallback_equal)
        return {
            "ok": bool(live.get("ok")) or include_static,
            "equal": equal,
            "live_ok": bool(live.get("ok")),
            "live_error": None if live.get("ok") else live.get("error"),
            "live_alerts": live_count,
            "static_label_sets": static_count,
            "source": "fallback",
        }

    prefer_keys = list(prefer) if prefer is not None else [
        "alertname",
        "service",
        "secondary_backend",
        "tenant",
    ]
    equal = suggest_equal_labels(
        label_sets, prefer=prefer_keys, min_count=min_count
    )
    return {
        "ok": True,
        "equal": equal,
        "live_ok": bool(live.get("ok")),
        "live_error": None if live.get("ok") else live.get("error"),
        "live_alerts": live_count,
        "static_label_sets": static_count,
        "label_sets": len(label_sets),
        "source": "live+static"
        if live_count and static_count
        else ("live" if live_count else "static"),
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
    inhibit_path: Optional[str] = None,
    merge_inhibit: bool = True,
) -> Dict[str, Any]:
    """Template → YAML (shell script / envsubst) + opsiyonel inhibit merge."""
    env = os.environ.copy()
    if slack_webhook is not None:
        env["RAG_ALERTMANAGER_SLACK_WEBHOOK"] = slack_webhook
    if webhook_url is not None:
        env["RAG_ALERTMANAGER_WEBHOOK_URL"] = webhook_url
    tmpl = template or env.get("ALERTMANAGER_TEMPLATE") or str(DEFAULT_TEMPLATE)
    out = output or env.get("ALERTMANAGER_OUTPUT") or str(DEFAULT_OUTPUT)
    env["ALERTMANAGER_TEMPLATE"] = tmpl
    env["ALERTMANAGER_OUTPUT"] = out
    if inhibit_path is not None:
        env["ALERTMANAGER_INHIBIT"] = inhibit_path
    # Shell merge kapalı; Python merge (dedupe) çalışır. Docker/direct sh hâlâ shell merge eder.
    if merge_inhibit:
        env["ALERTMANAGER_MERGE_INHIBIT"] = "0"
    if not Path(RENDER_SCRIPT).is_file():
        return {"ok": False, "error": "render_script_missing", "script": str(RENDER_SCRIPT)}
    proc = subprocess.run(
        ["sh", str(RENDER_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    result: Dict[str, Any] = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": out,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
    if proc.returncode == 0 and merge_inhibit:
        merged = merge_inhibit_rules_into_config(out, inhibit_path=inhibit_path)
        result["inhibit_merge"] = merged
        if not merged.get("ok"):
            result["ok"] = False
    return result


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


def extract_grafana_alert_labels(
    paths: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Grafana alerting/rules YAML dosyalarından label setlerini çıkar."""
    from pathlib import Path
    import re

    root = ROOT
    if paths is None:
        candidates: List[Path] = []
        for folder in ("alerting", "rules"):
            base = root / "grafana" / folder
            if base.is_dir():
                candidates.extend(sorted(base.glob("**/*.yaml")))
                candidates.extend(sorted(base.glob("**/*.yml")))
    else:
        candidates = []
        for p in paths:
            path = Path(p)
            if path.is_file():
                candidates.append(path)
            else:
                candidates.extend(sorted(Path().glob(str(p))))

    found: List[Dict[str, Any]] = []
    label_block = re.compile(
        r"labels:\s*\n((?:\s{2,}[A-Za-z0-9_]+\s*:\s*.+\n?)+)",
        re.MULTILINE,
    )
    title_re = re.compile(r"^\s*(?:title|alert|uid):\s*(.+)\s*$", re.MULTILINE)

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        titles = [m.group(1).strip().strip('"').strip("'") for m in title_re.finditer(text)]
        title_i = 0
        for m in label_block.finditer(text):
            block = m.group(1)
            labels: Dict[str, str] = {}
            for line in block.splitlines():
                line = line.strip()
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                labels[k.strip()] = v.strip().strip('"').strip("'")
            if not labels:
                continue
            name = titles[title_i] if title_i < len(titles) else path.stem
            title_i += 1
            found.append(
                {
                    "alertname": name,
                    "labels": labels,
                    "source_file": str(path),
                }
            )
    return found


def generate_inhibit_rules(
    label_sets: Sequence[Dict[str, Any]],
    *,
    severity_order: Sequence[str] = ("critical", "warning", "info"),
    equal_labels: Sequence[str] = ("alertname", "service"),
) -> List[Dict[str, Any]]:
    """Severity ladder inhibit kuralları üret (Alertmanager inhibit_rules)."""
    order = [s.strip().lower() for s in severity_order if str(s).strip()]
    equal = [e.strip() for e in equal_labels if str(e).strip()]
    # service bazında görülen severities
    by_service: Dict[str, set] = {}
    for item in label_sets:
        labels = item.get("labels") or {}
        if not isinstance(labels, dict):
            continue
        sev = str(labels.get("severity") or "").strip().lower()
        svc = str(labels.get("service") or "").strip() or "_default"
        if not sev:
            continue
        by_service.setdefault(svc, set()).add(sev)

    rules: List[Dict[str, Any]] = []
    seen = set()

    def _add(src: str, tgt: str, svc: Optional[str]) -> None:
        key = (src, tgt, svc or "*", tuple(equal))
        if key in seen:
            return
        seen.add(key)
        rule: Dict[str, Any] = {
            "source_matchers": [f"severity = {src}"],
            "target_matchers": [f"severity = {tgt}"],
            "equal": list(equal),
        }
        if svc and svc != "_default":
            rule["source_matchers"].append(f"service = {svc}")
            rule["target_matchers"].append(f"service = {svc}")
        rules.append(rule)

    # her service için ladder
    for svc, sevs in sorted(by_service.items()):
        present = [s for s in order if s in sevs]
        if len(present) >= 2:
            for i in range(len(present) - 1):
                for j in range(i + 1, len(present)):
                    _add(present[i], present[j], svc if svc != "_default" else None)
        else:
            # tek severity → canonical critical→warning (service varsa)
            _add("critical", "warning", svc if svc != "_default" else None)

    if not rules:
        _add("critical", "warning", None)
    return rules


def render_inhibit_rules_yaml(rules: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Generated by rag.cli alertmanager --generate-inhibit",
        "# Merge under Alertmanager `inhibit_rules:`",
        "inhibit_rules:",
    ]
    for rule in rules:
        lines.append("  - source_matchers:")
        for m in rule.get("source_matchers") or []:
            lines.append(f"      - {m}")
        lines.append("    target_matchers:")
        for m in rule.get("target_matchers") or []:
            lines.append(f"      - {m}")
        equal = rule.get("equal") or []
        if equal:
            quoted = ", ".join(f'"{e}"' for e in equal)
            lines.append(f"    equal: [{quoted}]")
    return "\n".join(lines) + "\n"


def canonicalize_inhibit_yaml(text: str) -> str:
    """Yorum/boş satırları atıp karşılaştırılabilir forma getir."""
    lines = []
    for line in (text or "").splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip() + ("\n" if lines else "")


def diff_inhibit_rules(
    current_path: Optional[str] = None,
    new_text: Optional[str] = None,
    *,
    new_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Mevcut inhibit vs aday metin/diff."""
    import difflib

    cur = Path(
        current_path
        or os.environ.get("ALERTMANAGER_INHIBIT", "").strip()
        or str(ROOT / "grafana" / "inhibit_rules.generated.yml")
    )
    if new_text is None:
        if not new_path or not Path(new_path).is_file():
            return {"ok": False, "error": "new_text_or_path_required"}
        new_text = Path(new_path).read_text(encoding="utf-8")
    old_text = cur.read_text(encoding="utf-8") if cur.is_file() else ""
    old_c = canonicalize_inhibit_yaml(old_text)
    new_c = canonicalize_inhibit_yaml(new_text or "")
    changed = old_c != new_c
    unified = "".join(
        difflib.unified_diff(
            old_c.splitlines(keepends=True),
            new_c.splitlines(keepends=True),
            fromfile=str(cur),
            tofile="candidate",
            n=3,
        )
    )
    return {
        "ok": True,
        "changed": changed,
        "current_path": str(cur),
        "current_exists": cur.is_file(),
        "unified_diff": unified,
        "old_bytes": len(old_c.encode("utf-8")),
        "new_bytes": len(new_c.encode("utf-8")),
    }


def apply_inhibit_equal_with_gate(
    *,
    paths: Optional[Sequence[str]] = None,
    output: Optional[str] = None,
    from_live: bool = True,
    api_url: Optional[str] = None,
    equal_labels: Sequence[str] = ("alertname", "service"),
    render_output: Optional[str] = None,
    require_amtool: bool = False,
    dry_run: bool = False,
    rollback_on_regression: Optional[bool] = None,
    backup_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Equal tune → inhibit üret → diff → render+check-config gate → opsiyonel yaz.
    Yazımdan sonra yeniden amtool; regresyonda önceki YAML'a rollback.
    """
    out = output or str(ROOT / "grafana" / "inhibit_rules.generated.yml")
    candidate = str(ROOT / "metadata" / "inhibit_rules.apply.yml")
    Path(candidate).parent.mkdir(parents=True, exist_ok=True)
    generated = write_inhibit_rules(
        paths=paths,
        output=candidate,
        equal_labels=equal_labels,
        from_live=from_live,
        api_url=api_url,
    )
    new_text = Path(candidate).read_text(encoding="utf-8")
    diff = diff_inhibit_rules(out, new_text)
    result: Dict[str, Any] = {
        "ok": True,
        "generated": generated,
        "diff": diff,
        "applied": False,
        "dry_run": dry_run,
        "output": out,
        "candidate": candidate,
        "rolled_back": False,
    }
    if not diff.get("changed"):
        result["reason"] = "unchanged"
        return result

    # Gate: render merged config + check-config
    rendered = render_output or str(ROOT / "metadata" / "alertmanager.apply.yml")
    render = render_alertmanager_config(
        output=rendered,
        inhibit_path=candidate,
        slack_webhook=os.environ.get(
            "RAG_ALERTMANAGER_SLACK_WEBHOOK",
            "https://hooks.slack.com/services/T/B/APPLY",
        ),
        webhook_url=os.environ.get(
            "RAG_ALERTMANAGER_WEBHOOK_URL",
            "http://127.0.0.1/hook",
        ),
    )
    result["render"] = {
        "ok": render.get("ok"),
        "output": render.get("output"),
        "error": render.get("stderr") or render.get("error"),
    }
    if not render.get("ok"):
        result["ok"] = False
        result["reason"] = "render_failed"
        return result

    checked = check_alertmanager_config(rendered)
    result["check"] = checked
    if require_amtool and checked.get("method") == "structural":
        result["ok"] = False
        result["reason"] = "amtool_required"
        return result
    if not checked.get("ok"):
        result["ok"] = False
        result["reason"] = "check_config_failed"
        return result

    if dry_run:
        result["reason"] = "dry_run_changed"
        return result

    do_rollback = rollback_on_regression
    if do_rollback is None:
        do_rollback = os.environ.get("INHIBIT_EQUAL_ROLLBACK", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    previous = Path(out).read_text(encoding="utf-8") if Path(out).is_file() else ""
    bak = backup_path or os.environ.get("INHIBIT_EQUAL_BACKUP_PATH", "").strip()
    if not bak:
        bak = str(ROOT / "metadata" / "inhibit_rules.generated.yml.bak")
    Path(bak).parent.mkdir(parents=True, exist_ok=True)
    Path(bak).write_text(previous, encoding="utf-8")
    result["backup"] = bak

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(new_text, encoding="utf-8")

    # Post-apply regression gate against the written file
    post_rendered = str(ROOT / "metadata" / "alertmanager.apply.post.yml")
    post_render = render_alertmanager_config(
        output=post_rendered,
        inhibit_path=out,
        slack_webhook=os.environ.get(
            "RAG_ALERTMANAGER_SLACK_WEBHOOK",
            "https://hooks.slack.com/services/T/B/APPLY",
        ),
        webhook_url=os.environ.get(
            "RAG_ALERTMANAGER_WEBHOOK_URL",
            "http://127.0.0.1/hook",
        ),
    )
    post_check = check_alertmanager_config(post_rendered)
    result["post_render"] = {
        "ok": post_render.get("ok"),
        "output": post_render.get("output"),
        "error": post_render.get("stderr") or post_render.get("error"),
    }
    result["post_check"] = post_check

    regress = (not post_render.get("ok")) or (not post_check.get("ok"))
    if require_amtool and post_check.get("method") == "structural":
        regress = True
    if regress and do_rollback:
        Path(out).write_text(previous, encoding="utf-8")
        result["ok"] = False
        result["applied"] = False
        result["rolled_back"] = True
        result["reason"] = "amtool_regression"
        return result

    result["applied"] = True
    result["reason"] = "applied"
    return result


def write_inhibit_rules(
    *,
    paths: Optional[Sequence[str]] = None,
    output: Optional[str] = None,
    equal_labels: Sequence[str] = ("alertname", "service"),
    severity_order: Sequence[str] = ("critical", "warning", "info"),
    from_live: bool = False,
    api_url: Optional[str] = None,
    tune_min_count: int = 1,
) -> Dict[str, Any]:
    equal = list(equal_labels)
    tune: Optional[Dict[str, Any]] = None
    if from_live:
        tune = tune_inhibit_equal_from_live(
            base_url=api_url,
            paths=paths,
            fallback_equal=equal_labels,
            min_count=tune_min_count,
            include_static=True,
        )
        if tune.get("equal"):
            equal = list(tune["equal"])
    label_sets: List[Dict[str, Any]] = list(extract_grafana_alert_labels(paths))
    if from_live:
        live = list_alerts(base_url=api_url)
        if live.get("ok"):
            label_sets.extend(extract_live_alert_labels(live.get("alerts") or []))
    rules = generate_inhibit_rules(
        label_sets, severity_order=severity_order, equal_labels=equal
    )
    text = render_inhibit_rules_yaml(rules)
    out = output or str(ROOT / "grafana" / "inhibit_rules.generated.yml")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    result: Dict[str, Any] = {
        "ok": True,
        "output": out,
        "rules": len(rules),
        "label_sets": len(label_sets),
        "equal": equal,
        "from_live": bool(from_live),
        "sources": sorted(
            {
                str(x.get("source_file") or x.get("source") or "")
                for x in label_sets
                if x.get("source_file") or x.get("source")
            }
        ),
    }
    if tune is not None:
        result["equal_tune"] = tune
    return result


DEFAULT_INHIBIT = ROOT / "grafana" / "inhibit_rules.generated.yml"


def _extract_inhibit_rule_blocks(text: str) -> List[str]:
    """inhibit_rules YAML içinden rule bloklarını ( '  - ...' ) ayıkla."""
    lines = text.splitlines()
    blocks: List[str] = []
    current: List[str] = []
    in_rules = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "inhibit_rules:" or stripped.startswith("inhibit_rules:"):
            in_rules = True
            continue
        if not in_rules:
            # generated file is only inhibit_rules; still accept rule starts
            if line.startswith("  - "):
                in_rules = True
            else:
                continue
        if line.startswith("  - "):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current and (line.startswith("    ") or stripped == ""):
            if stripped:
                current.append(line)
        elif current and not line.startswith(" ") and stripped:
            # left inhibit_rules section
            break
    if current:
        blocks.append("\n".join(current))
    return blocks


def merge_inhibit_rules_into_config(
    config_path: str,
    *,
    inhibit_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Rendered Alertmanager YAML'a generated inhibit_rules ekle (post-process).

    Template include yerine: envsubst sonrası text merge — generated dosya
    gitignore'da ve yoksa no-op. Mevcut kurallarla birebir aynı bloklar atlanır.
    """
    cfg = Path(config_path)
    inh = Path(
        inhibit_path
        or os.environ.get("ALERTMANAGER_INHIBIT", "").strip()
        or str(DEFAULT_INHIBIT)
    )
    if not cfg.is_file():
        return {"ok": False, "error": "config_missing", "config": str(cfg)}
    if not inh.is_file():
        return {
            "ok": True,
            "merged": False,
            "reason": "inhibit_missing",
            "config": str(cfg),
            "inhibit": str(inh),
            "added": 0,
        }

    cfg_text = cfg.read_text(encoding="utf-8")
    inh_text = inh.read_text(encoding="utf-8")
    new_blocks = _extract_inhibit_rule_blocks(inh_text)
    if not new_blocks:
        return {
            "ok": True,
            "merged": False,
            "reason": "no_rules",
            "config": str(cfg),
            "inhibit": str(inh),
            "added": 0,
        }

    existing = _extract_inhibit_rule_blocks(cfg_text)
    existing_norm = {b.strip() for b in existing}
    to_add = [b for b in new_blocks if b.strip() not in existing_norm]
    if not to_add:
        return {
            "ok": True,
            "merged": False,
            "reason": "already_present",
            "config": str(cfg),
            "inhibit": str(inh),
            "added": 0,
            "existing": len(existing),
        }

    if "inhibit_rules:" in cfg_text:
        # Append after last inhibit rule block / at end of inhibit_rules section
        addition = "\n" + "\n".join(to_add) + "\n"
        # Find inhibit_rules: and insert before next top-level key or EOF
        lines = cfg_text.splitlines(keepends=True)
        out_lines: List[str] = []
        in_inhibit = False
        inserted = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("inhibit_rules:"):
                in_inhibit = True
                out_lines.append(line)
                i += 1
                continue
            if in_inhibit:
                # still in section while indented or blank
                if stripped == "" or line.startswith(" ") or line.startswith("\t"):
                    out_lines.append(line)
                    i += 1
                    continue
                # top-level key → insert before it
                if not inserted:
                    if out_lines and not out_lines[-1].endswith("\n"):
                        out_lines[-1] += "\n"
                    out_lines.append(addition if addition.endswith("\n") else addition + "\n")
                    inserted = True
                in_inhibit = False
                out_lines.append(line)
                i += 1
                continue
            out_lines.append(line)
            i += 1
        if in_inhibit and not inserted:
            if out_lines and not str(out_lines[-1]).endswith("\n"):
                out_lines[-1] = str(out_lines[-1]) + "\n"
            out_lines.append(addition if addition.endswith("\n") else addition + "\n")
            inserted = True
        cfg.write_text("".join(out_lines), encoding="utf-8")
    else:
        block = "inhibit_rules:\n" + "\n".join(to_add) + "\n"
        with cfg.open("a", encoding="utf-8") as f:
            if not cfg_text.endswith("\n"):
                f.write("\n")
            f.write(block)

    return {
        "ok": True,
        "merged": True,
        "config": str(cfg),
        "inhibit": str(inh),
        "added": len(to_add),
        "existing": len(existing),
    }


def _structural_alertmanager_check(path: str) -> Dict[str, Any]:
    """amtool yoksa minimal yapısal doğrulama (route + receivers)."""
    import re

    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": "config_missing", "path": str(p), "method": "structural"}
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "path": str(p),
            "method": "structural",
        }
    if not text.strip():
        return {"ok": False, "error": "empty_config", "path": str(p), "method": "structural"}
    has_route = bool(
        re.search(r"(?m)^route\s*:", text)
        or text.lstrip().startswith("route:")
    )
    has_receivers = "receivers:" in text
    ok = has_route and has_receivers
    return {
        "ok": ok,
        "path": str(p),
        "method": "structural",
        "has_route": has_route,
        "has_receivers": has_receivers,
        "error": None if ok else "missing_route_or_receivers",
    }


def check_alertmanager_config(
    path: Optional[str] = None,
    *,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """amtool check-config (PATH/docker) veya yapısal fallback."""
    import shutil

    cfg = str(
        path
        or os.environ.get("ALERTMANAGER_OUTPUT", "").strip()
        or DEFAULT_OUTPUT
    )
    if not Path(cfg).is_file():
        return {"ok": False, "error": "config_missing", "path": cfg}

    amtool = shutil.which("amtool")
    if amtool:
        proc = subprocess.run(
            [amtool, "check-config", cfg],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "method": "amtool",
            "path": cfg,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "error": None if proc.returncode == 0 else "amtool_failed",
        }

    docker = shutil.which("docker")
    if docker:
        cfg_path = Path(cfg).resolve()
        proc = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "-v",
                f"{cfg_path}:/tmp/am.yml:ro",
                "prom/alertmanager:v0.27.0",
                "amtool",
                "check-config",
                "/tmp/am.yml",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "method": "docker_amtool",
            "path": cfg,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "error": None if proc.returncode == 0 else "docker_amtool_failed",
        }

    return _structural_alertmanager_check(cfg)
