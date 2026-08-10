"""Collab HTTP: aynı-origin Web Push SW + abonelik kaydı (PoC)."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

_http_started = False
_http_lock = threading.Lock()
_http_server: Optional[ThreadingHTTPServer] = None


def collab_http_public_base(
    *,
    public_host: Optional[str] = None,
    port: Optional[int] = None,
) -> str:
    try:
        from app.config import COLLAB_HTTP_PORT, COLLAB_WS_PUBLIC_HOST
    except ImportError:
        COLLAB_HTTP_PORT = 8766
        COLLAB_WS_PUBLIC_HOST = "localhost"
    host = (public_host or COLLAB_WS_PUBLIC_HOST or "localhost").strip() or "localhost"
    p = int(port if port is not None else COLLAB_HTTP_PORT)
    return f"http://{host}:{p}"


def handle_sw_js() -> Tuple[int, Dict[str, str], bytes]:
    from rag.collab_notify_push import load_service_worker_js

    body = (load_service_worker_js() or "").encode("utf-8")
    if not body:
        return 404, {"Content-Type": "text/plain; charset=utf-8"}, b"sw.js not found\n"
    headers = {
        "Content-Type": "application/javascript; charset=utf-8",
        "Service-Worker-Allowed": "/",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
    }
    return 200, headers, body


def _json_error(code: int, error: str) -> Tuple[int, Dict[str, str], bytes]:
    return (
        code,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps({"ok": False, "error": error}).encode("utf-8"),
    )


def handle_judge_ack_form(
    *,
    query: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """GET: Slack deep-link ack formu (token istemci tarafında girilir)."""
    qs = query or {}
    tenant = ((qs.get("tenant") or qs.get("tenant_id") or [""])[0] or "").strip()
    tenant_js = json.dumps(tenant)
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>Judge soft-fail ACK</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:480px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
label{{display:block;margin-top:12px;font-weight:600}}
input,textarea{{width:100%;padding:8px;box-sizing:border-box}}
button{{margin-top:14px;padding:8px 14px}}
#msg{{margin-top:12px;color:#333;white-space:pre-wrap}}
.tenant{{color:#555;font-size:0.9rem}}
</style></head><body>
<h1>Judge soft-fail acknowledge</h1>
<p>Token sunucu env <code>RAG_JUDGE_ACK_TOKEN</code> ile aynı olmalı.</p>
<p class="tenant" id="tenantLabel"></p>
<label>Actor <input id="actor" placeholder="oncall"/></label>
<label>Token <input id="token" type="password" placeholder="ack token"/></label>
<label>Note <textarea id="note" rows="3" placeholder="inceleme notu"></textarea></label>
<input type="hidden" id="tenant" value=""/>
<button id="go" type="button">Acknowledge</button>
<pre id="msg"></pre>
<script>
const TENANT = {tenant_js};
document.getElementById("tenant").value = TENANT || "";
const label = document.getElementById("tenantLabel");
if (TENANT) {{ label.textContent = "Tenant canvas: " + TENANT; }}
document.getElementById("go").onclick = async () => {{
  const msg = document.getElementById("msg");
  try {{
    const body = {{
      actor: (document.getElementById("actor").value || "").trim(),
      token: (document.getElementById("token").value || "").trim(),
      note: (document.getElementById("note").value || "").trim(),
      tenant_id: (document.getElementById("tenant").value || "").trim(),
      notify: "1"
    }};
    const r = await fetch("/judge/ack", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body)
    }});
    const data = await r.json();
    msg.textContent = JSON.stringify(data, null, 2);
  }} catch (e) {{
    msg.textContent = String(e && e.message ? e.message : e);
  }}
}};
</script>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def handle_judge_ack_export(
    *,
    query: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """GET: ack audit JSONL/CSV export (digest deep-link)."""
    from rag.judge_alert import export_judge_ack_audit

    qs = query or {}
    fmt = ((qs.get("format") or qs.get("fmt") or ["jsonl"])[0] or "jsonl").strip().lower()
    if fmt not in {"jsonl", "csv"}:
        fmt = "jsonl"
    limit_raw = (qs.get("limit") or [""])[0]
    limit = None
    try:
        if limit_raw:
            limit = int(limit_raw)
    except Exception:
        limit = 500
    if limit is None:
        limit = 500
    tenant = ((qs.get("tenant") or qs.get("tenant_id") or [""])[0] or "").strip() or None
    report = export_judge_ack_audit(fmt=fmt, limit=limit, tenant_id=tenant)
    text = report.get("text") or ""
    ctype = (
        "text/csv; charset=utf-8"
        if fmt == "csv"
        else "application/x-ndjson; charset=utf-8"
    )
    filename = f"judge_ack_audit.{ 'csv' if fmt == 'csv' else 'jsonl' }"
    headers = {
        "Content-Type": ctype,
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-cache",
        "X-Export-Count": str(report.get("count") or 0),
    }
    if tenant:
        headers["X-Export-Tenant"] = tenant
    return (
        200,
        headers,
        text.encode("utf-8"),
    )


def handle_judge_mute_snapshots(
    *,
    query: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """GET: HMAC-signed mute snapshot export download."""
    from rag.judge_alert import (
        read_judge_ack_digest_mute_export_file,
        verify_mute_export_signature,
    )

    qs = query or {}
    filename = ((qs.get("file") or qs.get("filename") or [""])[0] or "").strip()
    expires = ((qs.get("expires") or qs.get("exp") or [""])[0] or "").strip()
    sig = ((qs.get("sig") or qs.get("signature") or [""])[0] or "").strip()
    jti = ((qs.get("jti") or qs.get("token") or [""])[0] or "").strip() or None
    if not filename or not expires or not sig:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps(
                {"ok": False, "error": "file_expires_sig_required"},
                ensure_ascii=False,
            ).encode("utf-8"),
        )
    if not verify_mute_export_signature(filename, expires, sig, jti=jti):
        from rag.judge_alert import is_mute_export_revoked

        err = (
            "revoked"
            if is_mute_export_revoked(jti=jti, filename=filename)
            else "invalid_or_expired_signature"
        )
        return (
            403,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": err}, ensure_ascii=False).encode(
                "utf-8"
            ),
        )
    report = read_judge_ack_digest_mute_export_file(filename)
    if not report.get("ok"):
        return (
            404,
            {"Content-Type": "application/json"},
            json.dumps(report, ensure_ascii=False).encode("utf-8"),
        )
    kind = report.get("format") or "csv"
    ctype = (
        "text/csv; charset=utf-8"
        if kind == "csv"
        else "application/x-ndjson; charset=utf-8"
    )
    headers = {
        "Content-Type": ctype,
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "X-Mute-Export-Expires": str(expires),
    }
    if jti:
        headers["X-Mute-Export-Jti"] = jti
    return 200, headers, (report.get("text") or "").encode("utf-8")


def handle_ops_amtool_page() -> Tuple[int, Dict[str, str], bytes]:
    """GET: amtool / silence burn ops runbook page."""
    html = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>RAG amtool / silence burn</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a;background:linear-gradient(180deg,#f7fafc 0%,#eef2f7 100%);min-height:100vh}
h1{font-size:1.6rem;margin-bottom:0.25rem}
h2{font-size:1.15rem;margin-top:1.75rem}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.88rem}
pre{background:#0f172a;color:#e2e8f0;padding:12px 14px;overflow:auto}
.note{color:#475569;font-size:0.95rem}
a{color:#0f766e}
</style></head><body>
<h1>Inhibit equal — amtool &amp; silence burn</h1>
<p class="note">Runbook for <code>RagInhibitEqualCanarySilenceBurn</code> and Alertmanager equal apply gates.
Slack canary button + PagerDuty <code>custom_details.runbook_url</code> (<code>INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL</code>) point here.</p>
<p><a href="/ops/silence-burn">Silence burn runbook →</a></p>
<h2>amtool check-config</h2>
<pre>python -m rag.cli alertmanager --check-config
# veya
amtool check-config grafana/alertmanager.yml</pre>
<h2>List / inspect silences</h2>
<pre>python -m rag.cli alertmanager --list-silences
python -m rag.cli alertmanager --list-alerts</pre>
<h2>Equal apply gate</h2>
<pre>python -m rag.cli alertmanager --apply-equal --require-amtool
# CI: INHIBIT_EQUAL_REQUIRE_AMTOOL=1</pre>
<h2>Canary auto-silence knobs</h2>
<pre>INHIBIT_EQUAL_CANARY_AUTO_SILENCE=1
INHIBIT_EQUAL_CANARY_AUTO_SILENCE_DURATION=2h
# expiry webhook:
python -m rag.cli alertmanager --canary-silence-expiry</pre>
<p class="note">Dashboard: Grafana <code>rag-judge</code> → Inhibit equal canary silence / burn panels.</p>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def handle_ops_silence_burn_page() -> Tuple[int, Dict[str, str], bytes]:
    """GET: silence burn-rate triage runbook."""
    html = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>Silence burn runbook</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a;background:linear-gradient(180deg,#fff7ed 0%,#f8fafc 55%);min-height:100vh}
h1{font-size:1.6rem}
ol{padding-left:1.2rem}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.88rem}
pre{background:#111827;color:#f3f4f6;padding:12px 14px;overflow:auto}
a{color:#b45309}
</style></head><body>
<h1>Silence burn-rate runbook</h1>
<p>Alert: <code>RagInhibitEqualCanarySilenceBurn</code> — fail ratio on <code>rag_inhibit_equal_canary_silence_total</code>.</p>
<p>Slack canary button + PagerDuty/Opsgenie <code>details.runbook_url</code> (<code>INHIBIT_EQUAL_CANARY_PD_RUNBOOK_URL</code>) deep-link here.</p>
<ol>
<li>Confirm burn windows: <code>rag:inhibit_equal_canary_silence_fail_ratio:1h/6h</code> on Grafana rag-judge.</li>
<li>List silences and recent canary outcomes:</li>
</ol>
<pre>python -m rag.cli alertmanager --list-silences
python -m rag.cli alertmanager --check-config
# artifact: metadata/inhibit_equal_canary_silence.json</pre>
<ol start="3">
<li>If Alertmanager API rejects silences, fix auth/URL then re-run equal apply with amtool gate.</li>
<li>Expiry path: <code>--canary-silence-expiry</code> + <code>INHIBIT_EQUAL_CANARY_SILENCE_EXPIRY_WEBHOOK</code>.</li>
<li>Disable temporarily only if needed: <code>INHIBIT_EQUAL_CANARY_AUTO_SILENCE=0</code>.</li>
</ol>
<p><a href="/ops/amtool">amtool ops page →</a> · <a href="/judge/mute-export-revoke">mute export revoke UI →</a></p>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def handle_judge_mute_export_revoke_form(
    *,
    query: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """GET: admin UI to list/revoke mute export signed URLs."""
    from rag.judge_alert import list_mute_export_signed_urls

    qs = query or {}
    listed = list_mute_export_signed_urls(limit=50)
    rows_html = []
    for row in listed.get("links") or []:
        jti = str(row.get("jti") or "")
        flags = []
        if row.get("revoked"):
            flags.append("revoked")
        if row.get("expired"):
            flags.append("expired")
        flag_s = ",".join(flags) or "active"
        rows_html.append(
            "<tr>"
            f"<td><code>{jti}</code></td>"
            f"<td>{row.get('filename') or ''}</td>"
            f"<td>{row.get('expires') or ''}</td>"
            f"<td>{flag_s}</td>"
            f"<td><button type='button' data-jti='{jti}' class='rev'>Revoke</button></td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>jti</th><th>file</th><th>expires</th><th>status</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows_html) or '<tr><td colspan=5>no links</td></tr>'}</tbody></table>"
    )
    prefill = ((qs.get("jti") or qs.get("ref") or [""])[0] or "").strip()
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>Mute export revoke</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.45;background:linear-gradient(180deg,#f8fafc,#eef2ff)}}
label{{display:block;margin-top:12px;font-weight:600}}
input,textarea{{width:100%;padding:8px;box-sizing:border-box}}
button{{margin-top:12px;padding:8px 14px}}
table{{width:100%;border-collapse:collapse;margin-top:1rem;font-size:0.9rem}}
td,th{{border-bottom:1px solid #cbd5e1;padding:6px;text-align:left;vertical-align:top}}
#msg{{margin-top:12px;white-space:pre-wrap}}
code{{font-size:0.85rem}}
</style></head><body>
<h1>Mute export signed URL revoke</h1>
<p>Token = <code>RAG_JUDGE_ACK_TOKEN</code>. TTL sweep: <code>python -m rag.cli judge-ack-digest --sweep-mute-export-urls</code></p>
<label>JTI / filename <input id="ref" value="{prefill}" placeholder="jti or filename"/></label>
<label>Actor <input id="actor" placeholder="oncall"/></label>
<label>Token <input id="token" type="password" placeholder="ack token"/></label>
<label>Note <textarea id="note" rows="2" placeholder="why revoke"></textarea></label>
<button id="go" type="button">Revoke</button>
<pre id="msg"></pre>
{table}
<script>
document.querySelectorAll("button.rev").forEach(btn => {{
  btn.onclick = () => {{ document.getElementById("ref").value = btn.dataset.jti || ""; }};
}});
document.getElementById("go").onclick = async () => {{
  const msg = document.getElementById("msg");
  try {{
    const body = {{
      ref: (document.getElementById("ref").value || "").trim(),
      actor: (document.getElementById("actor").value || "").trim(),
      token: (document.getElementById("token").value || "").trim(),
      note: (document.getElementById("note").value || "").trim()
    }};
    const r = await fetch("/judge/mute-export-revoke", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body)
    }});
    const data = await r.json();
    msg.textContent = JSON.stringify(data, null, 2);
    if (data.ok) setTimeout(() => location.reload(), 600);
  }} catch (e) {{
    msg.textContent = String(e && e.message ? e.message : e);
  }}
}};
</script>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def handle_judge_mute_export_revoke(
    body: bytes,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """POST: revoke mute export signed URL (ack-token auth)."""
    expected = os.environ.get("RAG_JUDGE_ACK_TOKEN", "").strip()
    if not expected:
        return (
            503,
            {"Content-Type": "application/json"},
            json.dumps(
                {"ok": False, "error": "ack_token_not_configured"},
                ensure_ascii=False,
            ).encode("utf-8"),
        )
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_json"}).encode("utf-8"),
        )
    if not isinstance(data, dict):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "object_required"}).encode("utf-8"),
        )
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    token = str(data.get("token") or "").strip()
    if not token:
        token = hdrs.get("x-judge-ack-token", "").strip()
    if not token:
        auth = hdrs.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if token != expected:
        return (
            401,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_token"}).encode("utf-8"),
        )
    ref = str(data.get("ref") or data.get("jti") or data.get("filename") or "").strip()
    actor = str(data.get("actor") or data.get("by") or "").strip() or "admin-ui"
    note = str(data.get("note") or "")[:500]
    from rag.judge_alert import revoke_mute_export_signed_url

    result = revoke_mute_export_signed_url(ref, actor=actor, note=note or None)
    code = 200 if result.get("ok") else (404 if result.get("error") == "not_found" else 400)
    return (
        code,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps(result, ensure_ascii=False).encode("utf-8"),
    )


def handle_judge_slack_interactive(
    body: bytes,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """POST Slack interactivity (signed) → soft-fail ack."""
    from rag.judge_alert import (
        handle_slack_interactive_ack,
        parse_slack_interactive_payload,
        verify_slack_request_signature,
    )

    secret = os.environ.get("RAG_JUDGE_SLACK_SIGNING_SECRET", "").strip()
    if not secret:
        return _json_error(403, "slack_signing_secret_not_configured")
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    if not verify_slack_request_signature(
        body,
        timestamp=hdrs.get("x-slack-request-timestamp", ""),
        signature=hdrs.get("x-slack-signature", ""),
        signing_secret=secret,
    ):
        return _json_error(401, "invalid_slack_signature")

    payload = parse_slack_interactive_payload(body)
    if not payload:
        return _json_error(400, "invalid_payload")

    # Slack URL verification (Events API) — opsiyonel
    if payload.get("type") == "url_verification":
        challenge = str(payload.get("challenge") or "")
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps({"challenge": challenge}).encode("utf-8"),
        )

    result = handle_slack_interactive_ack(payload)
    if result.get("mode") == "modal_open":
        if result.get("ok"):
            # Slack expects empty 200 for successful modal open from block_actions
            return (200, {"Content-Type": "text/plain"}, b"")
        return _json_error(400, str(result.get("error") or "modal_open_failed"))

    if result.get("mode") == "modal_submit":
        if result.get("ok"):
            # clear modal; optional errors object for field validation
            return (
                200,
                {"Content-Type": "application/json"},
                b"{}",
            )
        err = str(result.get("error") or "ack_failed")
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "response_action": "errors",
                    "errors": {"ack_note_block": f"Ack failed: {err}"},
                }
            ).encode("utf-8"),
        )

    if result.get("mode") == "digest_reexport":
        text = str(result.get("text") or "Ack audit re-exported")
        resp = {
            "response_type": "ephemeral",
            "replace_original": False,
            "text": text[:2900],
        }
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(resp, ensure_ascii=False).encode("utf-8"),
        )

    if result.get("mode") in {"digest_mute", "digest_catch_up"}:
        text = str(result.get("text") or result.get("mode") or "Digest mute updated")
        if result.get("ok"):
            resp = {
                "response_type": "ephemeral",
                "replace_original": False,
                "text": text[:2900],
            }
            return (
                200,
                {"Content-Type": "application/json"},
                json.dumps(resp, ensure_ascii=False).encode("utf-8"),
            )
        err = str(result.get("error") or "digest_mute_failed")
        status = 429 if err == "rate_limited" else 400
        retry = result.get("retry_after_sec")
        body: Dict[str, Any] = {
            "response_type": "ephemeral",
            "text": text[:2900] if text else f"Digest action failed: {err}",
            "ok": False,
            "error": err,
        }
        headers_out = {"Content-Type": "application/json"}
        if err == "rate_limited":
            headers_out["Retry-After"] = str(int(float(retry or 1)))
            body["retry_after_sec"] = retry
        return (
            status,
            headers_out,
            json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )

    if result.get("ok"):
        text = (
            f"Soft-fail acknowledged by {result.get('state', {}).get('acknowledged_by')}"
            if isinstance(result.get("state"), dict)
            else "Soft-fail acknowledged"
        )
        if result.get("already"):
            text = "Already acknowledged"
        resp = {
            "response_type": "ephemeral",
            "replace_original": False,
            "text": text,
        }
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(resp, ensure_ascii=False).encode("utf-8"),
        )
    err = str(result.get("error") or "ack_failed")
    status = 409 if err == "no_active_soft_fail" else 400
    if err == "unknown_action":
        status = 400
    if err == "rate_limited":
        status = 429
        retry = result.get("retry_after_sec")
        text = f"Ack rate limited; retry after {retry}s" if retry is not None else "Ack rate limited"
        return (
            status,
            {"Content-Type": "application/json", "Retry-After": str(int(float(retry or 1)))},
            json.dumps(
                {
                    "response_type": "ephemeral",
                    "text": text,
                    "ok": False,
                    "error": err,
                    "retry_after_sec": retry,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )
    return (
        status,
        {"Content-Type": "application/json"},
        json.dumps(
            {
                "response_type": "ephemeral",
                "text": f"Ack failed: {err}",
                "ok": False,
                "error": err,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    )


def handle_judge_ack(body: bytes, *, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], bytes]:
    """POST JSON: {actor, note?, token?} — soft-fail manuel acknowledge."""
    expected = os.environ.get("RAG_JUDGE_ACK_TOKEN", "").strip()
    if not expected:
        return _json_error(403, "ack_token_not_configured")
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _json_error(400, "invalid_json")
    if not isinstance(data, dict):
        return _json_error(400, "object_required")
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    token = str(data.get("token") or "").strip()
    if not token:
        token = hdrs.get("x-judge-ack-token", "").strip()
    if not token:
        auth = hdrs.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if token != expected:
        return _json_error(401, "invalid_token")
    actor = str(data.get("actor") or data.get("by") or "").strip()
    note = str(data.get("note") or "")[:500]
    tenant_id = str(data.get("tenant_id") or data.get("tenant") or "").strip() or None
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
    notify = str(data.get("notify", "1")).strip().lower() not in {"0", "false", "no", "off"}
    from rag.judge_alert import acknowledge_judge_alert

    result = acknowledge_judge_alert(
        actor=actor,
        note=note,
        state_path=state_path,
        notify=notify,
        tenant_id=tenant_id,
    )
    if not result.get("ok") and result.get("error") == "actor_required":
        return _json_error(400, "actor_required")
    if not result.get("ok") and result.get("error") == "no_active_soft_fail":
        return _json_error(409, "no_active_soft_fail")
    payload = {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def handle_judge_alert_state() -> Tuple[int, Dict[str, str], bytes]:
    """GET soft-fail alert state (token gerekmez; soft_fail bayrağı + ack özeti)."""
    state_path = os.environ.get("RAG_JUDGE_ALERT_STATE", "").strip() or None
    from rag.judge_alert import load_judge_alert_state

    state = load_judge_alert_state(state_path)
    public = {
        "soft_fail": bool(state.get("soft_fail")),
        "source": state.get("source"),
        "acknowledged": bool(state.get("acknowledged")),
        "acknowledged_by": state.get("acknowledged_by"),
        "acknowledged_at": state.get("acknowledged_at"),
        "last_action": state.get("last_action"),
        "updated_at": state.get("updated_at"),
    }
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps({"ok": True, "state": public}, ensure_ascii=False).encode("utf-8"),
    )


def handle_webpush_register(body: bytes) -> Tuple[int, Dict[str, str], bytes]:
    """POST JSON: {username, subscription|token, label?} → register_device_token."""
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_json"}).encode("utf-8"),
        )
    if not isinstance(data, dict):
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "object_required"}).encode("utf-8"),
        )
    username = str(data.get("username") or "").strip()
    if not username:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "username_required"}).encode("utf-8"),
        )
    sub = data.get("subscription")
    token = data.get("token")
    if isinstance(sub, dict):
        token_raw = json.dumps(sub, ensure_ascii=False)
    elif isinstance(sub, str) and sub.strip():
        token_raw = sub.strip()
    elif isinstance(token, str) and token.strip():
        token_raw = token.strip()
    else:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "subscription_required"}).encode("utf-8"),
        )
    from rag.collab_notify_push import parse_webpush_subscription, register_device_token

    if parse_webpush_subscription(token_raw) is None:
        return (
            400,
            {"Content-Type": "application/json"},
            json.dumps({"ok": False, "error": "invalid_subscription"}).encode("utf-8"),
        )
    rec = register_device_token(
        username,
        token_raw,
        platform="webpush",
        label=str(data.get("label") or "browser")[:64] or "browser",
        device_name=str(data.get("device_name") or "Browser")[:64] or None,
        os_name=str(data.get("os_name") or "web")[:32] or None,
    )
    payload = {
        "ok": True,
        "username": username,
        "platform": "webpush",
        "token_preview": (rec or {}).get("token", "")[:24],
    }
    return (
        200,
        {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def handle_webpush_page(
    *,
    username: str = "",
    public_base: Optional[str] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    """Aynı-origin abonelik sayfası (SW + auto register)."""
    from rag.collab_notify_push import vapid_application_server_key, vapid_configured

    base = public_base or collab_http_public_base()
    vapid = vapid_application_server_key() or ""
    configured = vapid_configured() and bool(vapid)
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/><title>RAG Web Push</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:560px;margin:2rem auto;padding:0 1rem;line-height:1.45}}
button{{padding:8px 14px;margin-right:8px}}
#msg{{margin-top:10px;color:#444}}
code{{font-size:12px;word-break:break-all}}
</style></head><body>
<h1>Web Push abonelik</h1>
<p>Service Worker: <code>{base}/sw.js</code></p>
<p>Kullanıcı: <input id="user" value="{username}" placeholder="username"/></p>
<button id="sub" type="button" {"disabled" if not configured else ""}>Abone ol ve kaydet</button>
<pre id="msg"></pre>
<script>
const VAPID = {json.dumps(vapid)};
const SW_URL = {json.dumps(base + "/sw.js")};
const REG_URL = {json.dumps(base + "/webpush/register")};
function urlBase64ToUint8Array(base64String) {{
  const padding = "=".repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(base64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; ++i) arr[i] = raw.charCodeAt(i);
  return arr;
}}
document.getElementById("sub").onclick = async () => {{
  const msg = document.getElementById("msg");
  try {{
    if (!VAPID) throw new Error("VAPID yok");
    if (!window.isSecureContext && location.hostname !== "localhost")
      throw new Error("HTTPS veya localhost gerekli");
    const user = (document.getElementById("user").value || "").trim();
    if (!user) throw new Error("username gerekli");
    const perm = await Notification.requestPermission();
    if (perm !== "granted") throw new Error("izin reddedildi");
    const reg = await navigator.serviceWorker.register(SW_URL, {{ scope: "/" }});
    await navigator.serviceWorker.ready;
    let sub = await reg.pushManager.getSubscription();
    if (!sub) {{
      sub = await reg.pushManager.subscribe({{
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(VAPID)
      }});
    }}
    const r = await fetch(REG_URL, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ username: user, subscription: sub.toJSON(), label: "browser-pwa" }})
    }});
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || ("HTTP " + r.status));
    msg.textContent = "Kayıt OK: " + JSON.stringify(data);
  }} catch (e) {{
    msg.textContent = String(e && e.message ? e.message : e);
  }}
}};
</script>
</body></html>
"""
    return (
        200,
        {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"},
        html.encode("utf-8"),
    )


def _cors_preflight() -> Tuple[int, Dict[str, str], bytes]:
    return (
        204,
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Judge-Ack-Token",
            "Access-Control-Max-Age": "86400",
        },
        b"",
    )


class CollabHTTPHandler(BaseHTTPRequestHandler):
    public_base: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, code: int, headers: Dict[str, str], body: bytes) -> None:
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        code, headers, body = _cors_preflight()
        self._send(code, headers, body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/sw.js":
            self._send(*handle_sw_js())
            return
        if path in {"/webpush", "/webpush/"}:
            qs = parse_qs(parsed.query or "")
            user = (qs.get("user") or qs.get("username") or [""])[0]
            self._send(
                *handle_webpush_page(
                    username=user,
                    public_base=self.public_base or None,
                )
            )
            return
        if path in {"/judge/ack-form", "/judge/ack-ui"}:
            qs = parse_qs(parsed.query or "")
            self._send(*handle_judge_ack_form(query=qs))
            return
        if path == "/judge/ack-export":
            qs = parse_qs(parsed.query or "")
            self._send(*handle_judge_ack_export(query=qs))
            return
        if path == "/judge/mute-snapshots":
            qs = parse_qs(parsed.query or "")
            self._send(*handle_judge_mute_snapshots(query=qs))
            return
        if path in {
            "/judge/mute-export-revoke",
            "/judge/mute-export-revoke-form",
            "/judge/mute-export-revoke-ui",
        }:
            qs = parse_qs(parsed.query or "")
            self._send(*handle_judge_mute_export_revoke_form(query=qs))
            return
        if path in {"/ops/amtool", "/ops/amtool/"}:
            self._send(*handle_ops_amtool_page())
            return
        if path in {"/ops/silence-burn", "/ops/silence-burn/"}:
            self._send(*handle_ops_silence_burn_page())
            return
        if path in {"/judge/alert", "/judge/alert-state"}:
            self._send(*handle_judge_alert_state())
            return
        if path in {"/", "/health"}:
            body = json.dumps(
                {
                    "ok": True,
                    "service": "collab-http",
                    "routes": [
                        "/sw.js",
                        "/webpush",
                        "/webpush/register",
                        "/judge/ack",
                        "/judge/ack-form",
                        "/judge/ack-export",
                        "/judge/mute-snapshots",
                        "/judge/mute-export-revoke",
                        "/judge/slack-interactive",
                        "/judge/alert",
                        "/ops/amtool",
                        "/ops/silence-burn",
                        "/alertmanager",
                        "/hooks/dual-write-catch-up",
                        "/hooks/dual-write-dlq-quarantine",
                    ],
                }
            ).encode("utf-8")
            self._send(200, {"Content-Type": "application/json"}, body)
            return
        self._send(404, {"Content-Type": "text/plain"}, b"not found\n")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if path == "/webpush/register":
            self._send(*handle_webpush_register(raw))
            return
        if path == "/judge/ack":
            hdrs = {k: v for k, v in self.headers.items()}
            self._send(*handle_judge_ack(raw, headers=hdrs))
            return
        if path == "/judge/mute-export-revoke":
            hdrs = {k: v for k, v in self.headers.items()}
            self._send(*handle_judge_mute_export_revoke(raw, headers=hdrs))
            return
        if path in {"/judge/slack-interactive", "/slack/interactive"}:
            hdrs = {k: v for k, v in self.headers.items()}
            self._send(*handle_judge_slack_interactive(raw, headers=hdrs))
            return
        if path in {"/hooks/dual-write-dlq-quarantine"}:
            from rag.dual_write_webhook import handle_alertmanager_webhook_http

            hdrs = {k: v for k, v in self.headers.items()}
            self._send(
                *handle_alertmanager_webhook_http(
                    raw, headers=hdrs, mode="dlq_quarantine"
                )
            )
            return
        if path in {"/alertmanager", "/hooks/dual-write-catch-up", "/hooks/alertmanager"}:
            from rag.dual_write_webhook import handle_alertmanager_webhook_http

            hdrs = {k: v for k, v in self.headers.items()}
            self._send(*handle_alertmanager_webhook_http(raw, headers=hdrs))
            return
        self._send(404, {"Content-Type": "text/plain"}, b"not found\n")


def ensure_collab_http_server(
    *,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    enabled: bool = True,
    public_host: Optional[str] = None,
) -> bool:
    """Daemon thread ile collab HTTP (SW + register) başlatır."""
    global _http_started, _http_server
    if not enabled:
        return False
    try:
        from app.config import COLLAB_HTTP_PORT
    except ImportError:
        COLLAB_HTTP_PORT = 8766
    bind_port = int(port if port is not None else COLLAB_HTTP_PORT)
    with _http_lock:
        if _http_started:
            return True
        public = collab_http_public_base(public_host=public_host, port=bind_port)

        class _Handler(CollabHTTPHandler):
            public_base = public

        server = ThreadingHTTPServer((host, bind_port), _Handler)
        thread = threading.Thread(
            target=server.serve_forever,
            name="collab-http",
            daemon=True,
        )
        thread.start()
        _http_server = server
        _http_started = True
        return True


def run_collab_http_server(
    host: str = "0.0.0.0",
    port: Optional[int] = None,
    *,
    public_host: Optional[str] = None,
) -> None:
    """Bloklayarak HTTP sunucusu (test/CLI)."""
    try:
        from app.config import COLLAB_HTTP_PORT
    except ImportError:
        COLLAB_HTTP_PORT = 8766
    bind_port = int(port if port is not None else COLLAB_HTTP_PORT)
    public = collab_http_public_base(public_host=public_host, port=bind_port)

    class _Handler(CollabHTTPHandler):
        public_base = public

    server = ThreadingHTTPServer((host, bind_port), _Handler)
    print(f"Collab HTTP: {public}  (/sw.js, /webpush, /webpush/register, /judge/ack)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
