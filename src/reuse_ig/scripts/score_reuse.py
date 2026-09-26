"""Score every context of every reuse item under one reader. One output line per item:
{"test", "id", <context>: {logp, n_tokens, logp_by_kind, n_tokens_by_kind}}. `resume: true` appends
the items not yet scored (never overwrites)."""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("items_dir", "model", "backend", "output_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    resume = bool(config.get("resume", False))
    out_path = Path(config["output_dir"]) / "logprobs_reuse.jsonl"
    done = set()
    if resume and out_path.exists():
        done = {(json.loads(l)["test"], json.loads(l)["id"]) for l in open(out_path) if l.strip()}
        print(f"RESUME: {len(done)} items already scored")
    else:
        output_dir = init_directory(config["output_dir"], overwrite=overwrite); copy_config(config_path, output_dir)
    items = [json.loads(l) for l in open(Path(config["items_dir"]) / "items.jsonl")]
    items = [it for it in items if (it["test"], it["id"]) not in done]
    if debug:
        items = items[: config.get("debug_limit", 4)]
    if config["backend"] == "mlx":
        from src.analogy_ig.logprob import Scorer
        scorer = Scorer(config["model"])
    elif config["backend"] == "hf":
        from src.analogy_ig.logprob_hf import ScorerHF
        scorer = ScorerHF(config["model"], dtype=config.get("dtype", "bfloat16"))
    else:
        raise ValueError("FATAL: backend must be mlx or hf")
    print(f"scoring {len(items)} items with {config['model']}")
    with open(out_path, "a" if done else "w") as f:
        for it in tqdm(items):
            spans = [tuple(s) for s in it["spans"]]
            out = {"test": it["test"], "id": it["id"]}
            for name, msgs in it["contexts"].items():
                s = scorer.score(msgs, it["target"], spans); s.pop("token_logp", None); out[name] = s
            f.write(json.dumps(out) + "\n"); f.flush()
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); main(a.config_path, a.overwrite, a.debug)
