#!/bin/bash
# Run the multi-model log-prob sensitivity check on a Lambda Cloud box.
#   bash scripts/analogy_ig/lambda_run.sh <ip>          # sync, provision, start all models (detached)
#   bash scripts/analogy_ig/lambda_run.sh <ip> fetch    # pull logs + finished results back
# Requires LAMBDA_SSH_KEY_PATH and HF_TOKEN in ../comb-creat-eval/.env (not copied to the box
# except HF_TOKEN, which gated models need).
set -euo pipefail
IP="${1:?usage: lambda_run.sh <ip> [start|fetch] [script] [config glob]}"; MODE="${2:-start}"
# Which scorer and which configs the remote loop runs (defaults: the full scorer over the model configs).
SCRIPT="${3:-src/analogy_ig/scripts/score_logprob.py}"; GLOB="${4:-configs/analogy_ig/models/score_logprob_*.yaml}"
ENVF="$(dirname "$0")/../../../comb-creat-eval/.env"
PEM=$(grep '^LAMBDA_SSH_KEY_PATH' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'"); PEM="${PEM/#\~/$HOME}"
HFT=$(grep '^HF_TOKEN' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'")
SSH="ssh -i $PEM -o StrictHostKeyChecking=no -o ServerAliveInterval=30 ubuntu@$IP"
REMOTE=/home/ubuntu/analogy_ig

if [ "$MODE" = "fetch" ]; then
  mkdir -p logs/lambda
  rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" "ubuntu@$IP:$REMOTE/logs/" logs/lambda/
  rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --include='logprob_*/' --include='logprob_*/**' --exclude='*' \
    "ubuntu@$IP:$REMOTE/data/analogy_ig/dataset/downstream/" data/analogy_ig/dataset/downstream/
  echo "fetched"; exit 0
fi

echo "== sync code + dataset"
$SSH "mkdir -p $REMOTE/logs $REMOTE/data/analogy_ig/dataset"
rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --exclude '__pycache__' src configs scripts "ubuntu@$IP:$REMOTE/"
rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" data/analogy_ig/dataset/analogies.jsonl "ubuntu@$IP:$REMOTE/data/analogy_ig/dataset/"

echo "== provision (Lambda Stack python3 + torch; add transformers etc.)"
$SSH "pip install -q --disable-pip-version-check 'transformers==4.51.3' 'tokenizers<0.22' 'jinja2>=3.1.4' accelerate pyyaml tqdm pandas scipy sentencepiece protobuf huggingface_hub >/dev/null 2>&1; python3 -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__)'"

echo "== start detached run over all model configs"
$SSH "cd $REMOTE && cat > run_all.sh <<'RS'
#!/bin/bash
export HF_TOKEN=\"$HFT\"
export PYTHONPATH=$REMOTE
cd $REMOTE
for cfg in $GLOB; do
  tag=\$(basename \$cfg .yaml | sed 's/score_logprob_//')
  echo \"=== \$tag start \$(date)\" >> logs/run_all.log
  python3 $SCRIPT \$cfg > logs/\$tag.log 2>&1 || echo \"=== \$tag FAILED\" >> logs/run_all.log
  echo \"=== \$tag done \$(date)\" >> logs/run_all.log
done
echo ALL_DONE >> logs/run_all.log
RS
chmod +x run_all.sh && nohup ./run_all.sh > logs/nohup.out 2>&1 &
echo started"
