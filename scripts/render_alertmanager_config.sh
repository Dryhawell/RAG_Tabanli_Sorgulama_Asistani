#!/usr/bin/env sh
# Alertmanager config: template → rendered YAML (envsubst) + opsiyonel reload.
# Kullanım:
#   ./scripts/render_alertmanager_config.sh
#   ALERTMANAGER_RELOAD=1 ./scripts/render_alertmanager_config.sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
TEMPLATE="${ALERTMANAGER_TEMPLATE:-$ROOT/grafana/alertmanager.yml.template}"
OUTPUT="${ALERTMANAGER_OUTPUT:-$ROOT/grafana/alertmanager.rendered.yml}"
RELOAD_URL="${ALERTMANAGER_RELOAD_URL:-http://127.0.0.1:9093/-/reload}"

if [ ! -f "$TEMPLATE" ]; then
  echo "template yok: $TEMPLATE" >&2
  exit 1
fi

# Slack webhook yoksa güvenli placeholder (Alertmanager ayağa kalksın)
export RAG_ALERTMANAGER_SLACK_WEBHOOK="${RAG_ALERTMANAGER_SLACK_WEBHOOK:-https://hooks.slack.com/services/REPLACE/ME/PLEASE}"
export RAG_ALERTMANAGER_WEBHOOK_URL="${RAG_ALERTMANAGER_WEBHOOK_URL:-http://host.docker.internal:9999/alertmanager-default}"

if command -v envsubst >/dev/null 2>&1; then
  envsubst '${RAG_ALERTMANAGER_SLACK_WEBHOOK} ${RAG_ALERTMANAGER_WEBHOOK_URL}' \
    < "$TEMPLATE" > "$OUTPUT"
else
  # Minimal fallback: sed (sadece bilinen placeholder'lar)
  sed \
    -e "s|\${RAG_ALERTMANAGER_SLACK_WEBHOOK}|${RAG_ALERTMANAGER_SLACK_WEBHOOK}|g" \
    -e "s|\${RAG_ALERTMANAGER_WEBHOOK_URL}|${RAG_ALERTMANAGER_WEBHOOK_URL}|g" \
    "$TEMPLATE" > "$OUTPUT"
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
