#!/usr/bin/env bash
set -euo pipefail
# Watcher: check every 5 minutes for tokens file, then summarize and exit.

LOG=assets/experiments/logs/ufb_tokens_watch.log
echo "[WATCH] Started at $(date -Is)" | tee -a "$LOG"

find_latest_exp() {
  local d
  d=$(ls -1d assets/experiments/outputs/*_EXP_DPO_UFB_S500_8GPU_RERUN 2>/dev/null | sort | tail -n 1 || true)
  if [[ -z "$d" ]]; then
    d=$(ls -1d assets/experiments/outputs/*_EXP_DPO_UFB_S500_8GPU 2>/dev/null | sort | tail -n 1 || true)
  fi
  echo "$d"
}

summarize_tokens() {
  local f="$1"
  python - << 'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
lines = []
for line in p.open():
    line=line.strip()
    if not line.startswith('{'): continue
    try:
        lines.append(json.loads(line))
    except: pass
if not lines:
    print('[WATCH] No JSON lines to summarize')
    sys.exit(0)
best = min(lines, key=lambda r: r.get('entropy', 1e9))
def clean(tok):
    try:
        return tok.replace('Ġ',' ').replace('▁',' ').strip()
    except Exception:
        return tok
boost = [(clean(t), float(v)) for (t,v) in best.get('top', [])[:20]]
supp = [(clean(t), float(v)) for (t,v) in (best.get('suppressed') or [])[:20]]
out = {
  'best_j': best.get('j'),
  'entropy': best.get('entropy'),
  'boosted_top': boost,
  'suppressed_top': supp,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

while true; do
  EXP=$(find_latest_exp)
  if [[ -z "$EXP" ]]; then
    echo "[WATCH] No EXP dir yet; sleeping 300s" | tee -a "$LOG"
    sleep 300; continue
  fi
  TOKS="$EXP/artifacts/patchscope_tokens.jsonl"
  if [[ -f "$EXP/artifacts/delta.pt" && -f "$TOKS" ]]; then
    echo "[WATCH] Tokens ready at $(date -Is): $TOKS" | tee -a "$LOG"
    echo "[WATCH] Summary:" | tee -a "$LOG"
    summarize_tokens "$TOKS" | tee -a "$LOG"
    exit 0
  else
    echo "[WATCH] Not ready yet (delta: $( [[ -f "$EXP/artifacts/delta.pt" ]] && echo OK || echo MISSING ), tokens: $( [[ -f "$TOKS" ]] && echo OK || echo MISSING )); sleeping 300s" | tee -a "$LOG"
    sleep 300
  fi
done

