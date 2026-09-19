"""Controls for the target-path measurement, attached to an existing scoring run.

config `control`: "format" (abstract content-free path in the analogy prompt) or "named" (the source
concept is named but no path is shown).

For every record of a run, score the target path after a prompt with the analogy prompt's exact
shape but an abstract, content-free path in the source slot (A --r1--> B, B --r2--> C, ...) and
no source concept named. Writes one JSON line per record ({id, C_format}) as a downstream file of
the run; analyses join it by id.
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
    for k in ("upstream_dir", "dataset_dir", "model", "backend", "control"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, output_dir)
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    ids = [json.loads(l)["id"] for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    if debug:
        ids = ids[: config.get("debug_limit", 4)]
    if config["backend"] == "mlx":
        from src.analogy_ig.logprob import Scorer
        scorer = Scorer(config["model"])
    elif config["backend"] == "hf":
        from src.analogy_ig.logprob_hf import ScorerHF
        scorer = ScorerHF(config["model"], dtype=config.get("dtype", "bfloat16"))
    else:
        raise ValueError("FATAL: backend must be mlx or hf")
    control = config["control"]
    if control not in ("format", "named"):
        raise ValueError("FATAL: control must be 'format' or 'named'")
    print(f"scoring {control} control for {len(ids)} records with {config['model']}")
    with open(output_dir / f"logprobs_{control}.jsonl", "w") as f:
        for rid in tqdm(ids):
            rec = recs[rid]
            v, pv, n = rec["v"], rec["path_v"], rec["n_hops"]
            target, spans = R.render_path(pv)
            rels = R.relation_sequence(pv)
            msgs = R.ctx_align_format(v, n, rels) if control == "format" else R.ctx_align_named(v, rec["u"], n, rels)
            s = scorer.score(msgs, target, spans); s.pop("token_logp", None)
            f.write(json.dumps({"id": rid, f"C_{control}": s}) + "\n"); f.flush()
    print("wrote", output_dir / f"logprobs_{control}.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
