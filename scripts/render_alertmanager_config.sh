#!/usr/bin/env sh
# Alertmanager config: template → rendered YAML (envsubst) + opsiyonel reload.
# Generated inhibit_rules (ALERTMANAGER_INHIBIT) post-process merge edilir.
# Kullanım:
#   ./scripts/render_alertmanager_config.sh
#   ALERTMANAGER_RELOAD=1 ./scripts/render_alertmanager_config.sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
TEMPLATE="${ALERTMANAGER_TEMPLATE:-$ROOT/grafana/alertmanager.yml.template}"
OUTPUT="${ALERTMANAGER_OUTPUT:-$ROOT/grafana/alertmanager.rendered.yml}"
INHIBIT="${ALERTMANAGER_INHIBIT:-$ROOT/grafana/inhibit_rules.generated.yml}"
RELOAD_URL="${ALERTMANAGER_RELOAD_URL:-http://127.0.0.1:9093/-/reload}"

if [ ! -f "$TEMPLATE" ]; then
  echo "template yok: $TEMPLATE" >&2
  exit 1
fi

# Slack webhook yoksa güvenli placeholder (Alertmanager ayağa kalksın)
export RAG_ALERTMANAGER_SLACK_WEBHOOK="${RAG_ALERTMANAGER_SLACK_WEBHOOK:-https://hooks.slack.com/services/REPLACE/ME/PLEASE}"
export RAG_ALERTMANAGER_WEBHOOK_URL="${RAG_ALERTMANAGER_WEBHOOK_URL:-http://host.docker.internal:9999/alertmanager-default}"
# Ingest burn-rate → dual-write catch-up hook (yoksa default webhook)
export RAG_ALERTMANAGER_INGEST_WEBHOOK_URL="${RAG_ALERTMANAGER_INGEST_WEBHOOK_URL:-$RAG_ALERTMANAGER_WEBHOOK_URL}"
# Quarantine depth → dual-write DLQ quarantine hook
export RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL="${RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL:-http://host.docker.internal:8766/hooks/dual-write-dlq-quarantine}"

if command -v envsubst >/dev/null 2>&1; then
  envsubst '${RAG_ALERTMANAGER_SLACK_WEBHOOK} ${RAG_ALERTMANAGER_WEBHOOK_URL} ${RAG_ALERTMANAGER_INGEST_WEBHOOK_URL} ${RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL}' \
    < "$TEMPLATE" > "$OUTPUT"
else
  # Minimal fallback: sed (sadece bilinen placeholder'lar)
  sed \
    -e "s|\${RAG_ALERTMANAGER_SLACK_WEBHOOK}|${RAG_ALERTMANAGER_SLACK_WEBHOOK}|g" \
    -e "s|\${RAG_ALERTMANAGER_WEBHOOK_URL}|${RAG_ALERTMANAGER_WEBHOOK_URL}|g" \
    -e "s|\${RAG_ALERTMANAGER_INGEST_WEBHOOK_URL}|${RAG_ALERTMANAGER_INGEST_WEBHOOK_URL}|g" \
    -e "s|\${RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL}|${RAG_ALERTMANAGER_INGEST_DLQ_QUARANTINE_WEBHOOK_URL}|g" \
    "$TEMPLATE" > "$OUTPUT"
fi

# Post-process: merge generated inhibit_rules (yoksa no-op).
# Template include yerine — generated dosya gitignore'da.
# Python render_alertmanager_config ALERTMANAGER_MERGE_INHIBIT=0 ile shell merge'i kapatır.
if [ "${ALERTMANAGER_MERGE_INHIBIT:-1}" != "0" ] && [ -f "$INHIBIT" ]; then
  # Yorum ve inhibit_rules: başlığını at; rule bloklarını ekle
  awk '
    /^[[:space:]]*#/ { next }
    /^inhibit_rules:[[:space:]]*$/ { next }
    /^inhibit_rules:/ { next }
    /^[[:space:]]+- / { print; next }
    /^[[:space:]]{2,}/ { print; next }
  ' "$INHIBIT" >> "$OUTPUT"
  echo "inhibit_merged: $INHIBIT"
fi

echo "rendered: $OUTPUT"

if [ "${ALERTMANAGER_RELOAD:-0}" = "1" ] || [ "${ALERTMANAGER_RELOAD:-}" = "true" ]; then
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS -X POST "$RELOAD_URL" >/dev/null; then
      echo "reloaded: $RELOAD_URL"
    else
      echo "reload failed: $RELOAD_URL" >&2
      exit 2
    fi
  else
    echo "curl yok; reload atlandı ($RELOAD_URL)" >&2
  fi
fi
