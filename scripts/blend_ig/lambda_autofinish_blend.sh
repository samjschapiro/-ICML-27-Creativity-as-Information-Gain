#!/bin/bash
# Unattended finisher for the blend ladder: wait for ALL_DONE, fetch, verify every GPU reader's file
# has as many lines as the dataset has blends, terminate the instance. Hard cap so an idle box
# cannot bill.
set -u
IP="${1:?ip}"; INST="${2:?instance id}"; EXPECT=$(wc -l < data/blend_ig/dataset/blends.jsonl); MAX_WAIT_MIN=150
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
ENVF="$ROOT/../comb-creat-eval/.env"
PEM=$(grep '^LAMBDA_SSH_KEY_PATH' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'"); PEM="${PEM/#\~/$HOME}"
KEY=$(grep '^LAMBDA_CLOUD_API_KEY' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'")
SSH="ssh -i $PEM -o StrictHostKeyChecking=no -o ConnectTimeout=20 ubuntu@$IP"
start=$(date +%s); log() { echo "$(date '+%H:%M:%S') $*"; }
terminate() { curl -s -u "$KEY:" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$INST\"]}" | head -c 200; echo; log "terminate requested for $INST"; }
while true; do
  status=$($SSH 'tail -1 ~/blend_ig/logs/run_all.log 2>/dev/null' 2>/dev/null || echo "ssh-failed")
  elapsed=$(( ($(date +%s) - start) / 60 )); log "status='$status' elapsed=${elapsed}m"
  if [ "$status" = "ALL_DONE" ]; then
    bash scripts/blend_ig/lambda_run.sh "$IP" fetch; ok=1
    for tag in llama31_8b mistral7b_v03 gemma2_9b qwen25_14b olmo2_7b; do
      f="data/blend_ig/dataset/downstream/blendblock_$tag/logprobs_blend_block.jsonl"
      n=$(wc -l < "$f" 2>/dev/null || echo 0); log "  $tag: $n"; [ "$n" -eq "$EXPECT" ] || ok=0
    done
    if [ "$ok" -eq 1 ]; then terminate; log "DONE_OK"; exit 0; else log "DONE_INCOMPLETE (instance left running)"; exit 1; fi
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_MIN" ]; then bash scripts/blend_ig/lambda_run.sh "$IP" fetch; terminate; log "DONE_TIMEOUT"; exit 2; fi
  sleep 120
done
