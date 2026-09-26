#!/bin/bash
# Run the reuse items under the five GPU readers on a Lambda Cloud box.
#   bash scripts/reuse_ig/lambda_run.sh <ip>          # sync, provision, start all readers (detached)
#   bash scripts/reuse_ig/lambda_run.sh <ip> fetch    # pull logs + finished results back
set -euo pipefail
IP="${1:?usage: lambda_run.sh <ip> [start|fetch]}"; MODE="${2:-start}"
SCRIPT="src/reuse_ig/scripts/score_reuse.py"; GLOB="configs/reuse_ig/models/score_reuse_*.yaml"
ENVF="$(dirname "$0")/../../../comb-creat-eval/.env"
PEM=$(grep '^LAMBDA_SSH_KEY_PATH' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'"); PEM="${PEM/#\~/$HOME}"
HFT=$(grep '^HF_TOKEN' "$ENVF" | cut -d= -f2- | tr -d '"' | tr -d "'")
SSH="ssh -i $PEM -o StrictHostKeyChecking=no -o ServerAliveInterval=30 ubuntu@$IP"
REMOTE=/home/ubuntu/reuse_ig

if [ "$MODE" = "fetch" ]; then
  mkdir -p logs/lambda_reuse data/reuse_ig/items/downstream
  rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" "ubuntu@$IP:$REMOTE/logs/" logs/lambda_reuse/
  rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --include='reuse_*/' --include='reuse_*/**' --exclude='*' \
    "ubuntu@$IP:$REMOTE/data/reuse_ig/items/downstream/" data/reuse_ig/items/downstream/
  echo "fetched"; exit 0
fi

echo "== sync code + items"
$SSH "mkdir -p $REMOTE/logs $REMOTE/data/reuse_ig/items"
rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" --exclude '__pycache__' src configs scripts "ubuntu@$IP:$REMOTE/"
rsync -az -e "ssh -i $PEM -o StrictHostKeyChecking=no" data/reuse_ig/items/items.jsonl "ubuntu@$IP:$REMOTE/data/reuse_ig/items/"

echo "== provision"
$SSH "pip install -q --disable-pip-version-check 'transformers==4.51.3' 'tokenizers<0.22' 'jinja2>=3.1.4' accelerate pyyaml tqdm pandas scipy sentencepiece protobuf huggingface_hub >/dev/null 2>&1; python3 -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"transformers\", transformers.__version__)'"

echo "== start detached run over all reader configs"
$SSH "cd $REMOTE && cat > run_all.sh <<'RS'
#!/bin/bash
export HF_TOKEN=\"$HFT\"
export PYTHONPATH=$REMOTE
cd $REMOTE
for cfg in $GLOB; do
  tag=\$(basename \$cfg .yaml | sed 's/score_reuse_//')
  echo \"=== \$tag start \$(date)\" >> logs/run_all.log
  python3 $SCRIPT \$cfg > logs/\$tag.log 2>&1 || echo \"=== \$tag FAILED\" >> logs/run_all.log
  echo \"=== \$tag done \$(date)\" >> logs/run_all.log
done
echo ALL_DONE >> logs/run_all.log
RS
chmod +x run_all.sh && nohup ./run_all.sh > logs/nohup.out 2>&1 &
echo started"
