"""Score the whole description of each blend under the six inputs of the ladder.

Scored block: every triple of the blend as one line (see render.render_block). Inputs: skeleton,
input u, input v, inputs, vacuous schema (fixed control), generic space. Writes one line per blend:
{"id", "n_triples", "tags", <input>: {logp, n_tokens, logp_by_kind, n_tokens_by_kind}} where the
by-kind keys are '<line index>:<entity|relation>' or 'sep', so per-triple sums are recoverable.
`resume: true` appends the blends not yet in an existing output file (never overwrites).
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.blend_ig import render as R


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("dataset_dir", "model", "backend", "output_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    resume = bool(config.get("resume", False))
    out_path = Path(config["output_dir"]) / "logprobs_blend_block.jsonl"
    done_ids: set[str] = set()
    if resume and out_path.exists():
        output_dir = Path(config["output_dir"])
        done_ids = {json.loads(l)["id"] for l in open(out_path) if l.strip()}
        print(f"RESUME: {len(done_ids)} blends already scored; appending the rest")
    else:
        output_dir = init_directory(config["output_dir"], overwrite=overwrite)
        copy_config(config_path, output_dir)
    recs = [json.loads(l) for l in open(Path(config["dataset_dir"]) / "blends.jsonl")]
    recs = [r for r in recs if r["id"] not in done_ids]
    if debug:
        recs = recs[: config.get("debug_limit", 4)]
    if config["backend"] == "mlx":
        from src.analogy_ig.logprob import Scorer
        scorer = Scorer(config["model"])
    elif config["backend"] == "hf":
        from src.analogy_ig.logprob_hf import ScorerHF
        scorer = ScorerHF(config["model"], dtype=config.get("dtype", "bfloat16"))
    else:
        raise ValueError("FATAL: backend must be mlx or hf")
    print(f"scoring blend blocks for {len(recs)} blends with {config['model']}")
    with open(out_path, "a" if done_ids else "w") as f:
        for rec in tqdm(recs):
            target, spans = R.render_block(rec["triples"])
            out = {"id": rec["id"], "n_triples": rec["n_triples"], "tags": rec["tags"]}
            for name, msgs in R.all_inputs(rec).items():
                s = scorer.score(msgs, target, spans); s.pop("token_logp", None)
                out[name] = s
            f.write(json.dumps(out) + "\n"); f.flush()
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
