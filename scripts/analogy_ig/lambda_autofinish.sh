#!/bin/bash
# Unattended finisher for a Lambda scoring run: poll until the remote loop reports ALL_DONE,
# fetch results, verify every reader file is complete, then terminate the instance via the API.
#   nohup bash scripts/analogy_ig/lambda_autofinish.sh <ip> <instance_id> > logs/autofinish.log 2>&1 &
# Safety: terminates only after a complete fetch (977 lines per reader), or after MAX_WAIT_MIN
# regardless (with whatever was fetched) so an unattended box cannot bill indefinitely.
set -u
IP="${1:?ip}"; INST="${2:?instance id}"; EXPECT=977; MAX_WAIT_MIN=240
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
ENVF="$ROOT/../comb-creat-eval/.env"
PEM=$(grep '^LAMBDA_SSH_KEY_PATH' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'"); PEM="${PEM/#\~/$HOME}"
KEY=$(grep '^LAMBDA_CLOUD_API_KEY' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'")
SSH="ssh -i $PEM -o StrictHostKeyChecking=no -o ConnectTimeout=20 ubuntu@$IP"
start=$(date +%s)
log() { echo "$(date '+%H:%M:%S') $*"; }
terminate() {
  curl -s -u "$KEY:" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$INST\"]}" | head -c 300; echo
  log "terminate requested for $INST"
}
while true; do
  status=$($SSH 'tail -1 ~/analogy_ig/logs/run_all.log 2>/dev/null' 2>/dev/null || echo "ssh-failed")
  counts=$($SSH 'wc -l ~/analogy_ig/data/analogy_ig/dataset/downstream/*/logprobs.jsonl 2>/dev/null | grep -v total | awk "{print \$1}" | tr "\n" " "' 2>/dev/null)
  elapsed=$(( ($(date +%s) - start) / 60 ))
  log "status='$status' counts='$counts' elapsed=${elapsed}m"
  if [ "$status" = "ALL_DONE" ]; then
    log "remote loop finished; fetching"
    bash scripts/analogy_ig/lambda_run.sh "$IP" fetch
    ok=1
    for tag in gemma2_9b llama31_8b mistral7b_v03 olmo2_7b qwen25_14b; do
      f="data/analogy_ig/dataset/downstream/logprob_${tag}_projdir/logprobs.jsonl"
      n=$(wc -l < "$f" 2>/dev/null || echo 0); log "  $tag: $n lines"
      [ "$n" -eq "$EXPECT" ] || ok=0
    done
    if [ "$ok" -eq 1 ]; then log "all readers complete; terminating"; terminate; log "DONE_OK"; exit 0
    else log "INCOMPLETE fetch; instance left running for inspection"; log "DONE_INCOMPLETE"; exit 1; fi
  fi
  if [ "$elapsed" -ge "$MAX_WAIT_MIN" ]; then
    log "max wait reached; fetching whatever exists and terminating"
    bash scripts/analogy_ig/lambda_run.sh "$IP" fetch; terminate; log "DONE_TIMEOUT"; exit 2
  fi
  sleep 300
done
