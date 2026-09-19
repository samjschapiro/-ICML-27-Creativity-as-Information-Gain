"""Invention measurement framed like the alignment one.

Task: describe the novel concept using a given relational skeleton (relations, fact count and the
concept's slot fixed; entities to supply). Scored block: the whole description M[Phi], joint
log-probability. Three inputs: structure only (task), analogy instruction (name the source concept),
analogy content (source concept's facts plus the two paths with correspondences). Attached to an
existing run (ids and projection direction come from its logprobs.jsonl); writes one line per record
with the three joint log-probabilities and per-kind sums.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig import render as R


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("upstream_dir", "dataset_dir", "model", "backend"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, output_dir)
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    rows = [json.loads(l) for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    rows = [r for r in rows if r.get("type3") and "images" in r["type3"]]
    if debug:
        rows = rows[: config.get("debug_limit", 4)]
    if config["backend"] == "mlx":
        from src.analogy_ig.logprob import Scorer
        scorer = Scorer(config["model"])
    elif config["backend"] == "hf":
        from src.analogy_ig.logprob_hf import ScorerHF
        scorer = ScorerHF(config["model"], dtype=config.get("dtype", "bfloat16"))
    else:
        raise ValueError("FATAL: backend must be mlx or hf")
    print(f"scoring invention blocks for {len(rows)} records with {config['model']}")
    with open(output_dir / "logprobs_invention_block.jsonl", "w") as f:
        for row in tqdm(rows):
            rec = recs[row["id"]]
            src = [im_s for im_s in (p["source"] for p in rec["projection"]) if isinstance(im_s, list) and len(im_s) == 3]
            img = [im for im in (p["image"] for p in rec["projection"]) if isinstance(im, list) and len(im) == 3]
            if not img or len(src) != len(img):
                continue
            direction = row.get("projection_direction", "assumed_u_to_v")
            if direction == "v_to_u":
                sa, ta, sp, tp = rec["v"], rec["u"], rec["path_v"], rec["path_u"]
            else:
                sa, ta, sp, tp = rec["u"], rec["v"], rec["path_u"], rec["path_v"]
            phi, h = rec["projected"], rec["invention"]
            task = R.invention_task(h, ta, img)
            target, spans = R.render_path(img)
            conds = {
                "structure": R.ctx_invention_block(task, None),
                "instruction": R.ctx_invention_block(task, R.inv_block_instruction(phi, sa)),
                "mapping": R.ctx_invention_block(task, R.inv_block_mapping(phi, sa, ta, sp, tp)),
                "content": R.ctx_invention_block(task, R.inv_block_content(phi, sa, ta, src, sp, tp)),
            }
            out = {"id": row["id"], "n_facts": len(img), "projection_direction": direction}
            for name, msgs in conds.items():
                s = scorer.score(msgs, target, spans); s.pop("token_logp", None)
                out[name] = s
            f.write(json.dumps(out) + "\n"); f.flush()
    print("wrote", output_dir / "logprobs_invention_block.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
