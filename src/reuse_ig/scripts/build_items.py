"""Build every scoring item of the reuse track once, so all readers score identical prompts.

Writes items.jsonl: {"test", "id", "target", "spans", "contexts": {name: messages}, "meta"}.
Baseline inputs that the analogy and blend tracks already scored (relations only, source path,
told-it-blends, generic space) are not rebuilt; the analysis pulls them by id from those runs.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.reuse_ig import render as R


def _strict(r):
    return r["U_an"] and r["integration"] and r["integration_unanimous"] and r["utility"] and r["utility_unanimous"]


def _direction(rec, projdir_rows):
    d = projdir_rows.get(rec["id"], {}).get("projection_direction", "assumed_u_to_v")
    if d == "v_to_u":
        return rec["v"], rec["u"], rec["path_v"], rec["path_u"]
    return rec["u"], rec["v"], rec["path_u"], rec["path_v"]


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("analogy_dataset", "blend_dataset", "projdir_run", "demos", "other_pair_prompt", "output_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite); copy_config(config_path, out)
    A = [json.loads(l) for l in open(Path(config["analogy_dataset"]) / "analogies.jsonl")]
    B = [json.loads(l) for l in open(Path(config["blend_dataset"]) / "blends.jsonl")]
    proj = {}
    for l in open(Path(config["projdir_run"]) / "logprobs.jsonl"):
        r = json.loads(l); proj[r["id"]] = r
    byid = {r["id"]: r for r in A}
    demos = [byid[i] for i in config["demos"]]
    for d in demos:
        if not _strict(d):
            raise ValueError(f"FATAL: demonstration {d['id']} is not a strict-valid analogy")
    demo_pairs = {d["prompt_id"] for d in demos}
    other_prompt = config["other_pair_prompt"]              # e.g. "22": E22 analogy, F22 blend
    other_analogy = demos[0]
    if other_analogy["prompt_id"] != f"E{other_prompt}":
        raise ValueError("FATAL: the first demonstration must be the other-pair analogy")
    valid_A = [r for r in A if r["U_an"]]
    valid_B = [r for r in B if r["V_bl"]]

    # deterministic same-pair partners: alphabetically first model other than the item's own
    def partner_analogy(prompt_num, own_model):
        c = sorted((r for r in A if r["prompt_id"] == f"E{prompt_num}" and _strict(r) and r["model"] != own_model), key=lambda r: r["model"])
        return c[0] if c else None

    def partner_blend(prompt_num, own_model):
        c = sorted((r for r in B if r["prompt_id"] == f"F{prompt_num}" and r["V_bl"] and r["model"] != own_model), key=lambda r: r["model"])
        return c[0] if c else None

    other_blend = partner_blend(other_prompt, "")
    items, counts = [], Counter()

    def add(test, rid, built, meta):
        items.append({"test": test, "id": rid, "target": built["target"], "spans": built["spans"],
                      "contexts": built["contexts"], "meta": meta}); counts[test] += 1

    for r in valid_A:
        num = r["prompt_id"][1:]
        # 1a held-out hop
        if len(r["path_v"]) >= 3:
            add("heldout_hop", r["id"], R.heldout_hop(r["u"], r["v"], r["path_u"], r["path_v"]), {"n_hops": len(r["path_v"])})
        # 1b held-out projected fact
        src = [p["source"] for p in r["projection"] if isinstance(p["source"], list) and len(p["source"]) == 3]
        img = [p["image"] for p in r["projection"] if isinstance(p["image"], list) and len(p["image"]) == 3]
        if len(img) >= 2 and len(src) == len(img):
            sa, ta, sp, tp = _direction(r, proj)
            add("heldout_fact", r["id"], R.heldout_fact(r["invention"], ta, r["projected"], sa, sp, tp, src, img), {"n_facts": len(img)})
        # 2a within-task reuse (demonstration pairs excluded)
        if r["prompt_id"] not in demo_pairs:
            add("within_task", r["id"], R.within_task(r["v"], r["u"], r["path_u"], r["path_v"], demos[:1], demos[:3]), {})
        # 2b blend -> analogy
        pb = partner_blend(num, r["model"])
        if pb is not None and other_blend is not None:
            add("blend_to_analogy", r["id"], R.blend_to_analogy(r["v"], r["u"], r["path_u"], r["path_v"], pb["generic_space"],
                                                                 other_blend["generic_space"], (other_blend["u"], other_blend["v"])),
                {"partner_blend": pb["id"]})
    for b in valid_B:
        num = b["prompt_id"][1:]
        if num == other_prompt:
            continue
        pa = partner_analogy(num, b["model"])
        if pa is None:
            continue
        add("analogy_to_blend", b["id"], R.analogy_to_blend(b, pa, other_analogy), {"partner_analogy": pa["id"], "tags": b["tags"]})

    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    summary = {"n_items": len(items), "by_test": dict(counts), "demos": [d["id"] for d in demos],
               "other_pair": other_prompt, "n_contexts": sum(len(it["contexts"]) for it in items)}
    json.dump(summary, open(out / "summary.json", "w"), indent=2); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); main(a.config_path, a.overwrite, a.debug)
