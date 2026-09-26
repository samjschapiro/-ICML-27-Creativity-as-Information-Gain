"""Cross-reader analysis of the reuse tests. Baseline inputs come from the existing runs (by id):
relations only (C_rho) and source path (C1) from the analogy run, told-it-blends (inputs) and
generic space (generic) from the blend run. Reports, per reader and averaged over readers, the
mean log-probability under each input and paired gains with 95% CI, share positive, Wilcoxon p."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config

TESTS = {
    "heldout_hop": ["target_so_far", "instruction", "source_path"],
    "heldout_fact": ["known_facts", "instruction", "mapping", "content"],
    "within_task": ["relations", "relations_demo1", "relations_demo3", "source", "source_demo1", "source_demo3"],
    "blend_to_analogy": ["relations", "relations_generic", "relations_other_generic", "source", "source_generic"],
    "analogy_to_blend": ["inputs", "inputs_analogy", "generic", "inputs_analogy_generic", "inputs_other_analogy"],
}
GAINS = {
    "heldout_hop": [("instruction", "target_so_far"), ("source_path", "target_so_far"), ("source_path", "instruction")],
    "heldout_fact": [("instruction", "known_facts"), ("mapping", "known_facts"), ("mapping", "instruction"), ("content", "mapping")],
    "within_task": [("relations_demo1", "relations"), ("relations_demo3", "relations"), ("source_demo1", "source"),
                    ("source_demo3", "source"), ("source", "relations")],
    "blend_to_analogy": [("relations_generic", "relations"), ("relations_other_generic", "relations"),
                         ("source_generic", "source"), ("source", "relations")],
    "analogy_to_blend": [("inputs_analogy", "inputs"), ("inputs_other_analogy", "inputs"), ("generic", "inputs"),
                         ("inputs_analogy_generic", "generic"), ("inputs_analogy_generic", "inputs_analogy")],
}


def _paired(d):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "frac_positive": float((d > 0).mean()), "n": int(len(d)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}


def _frame(reader_cfg: dict) -> pd.DataFrame:
    rows = {}
    for l in open(Path(reader_cfg["reuse"]) / "logprobs_reuse.jsonl"):
        r = json.loads(l)
        row = {k: v["logp"] for k, v in r.items() if isinstance(v, dict)}
        rows[(r["test"], r["id"])] = row
    ana = {}
    for l in open(Path(reader_cfg["analogy_run"]) / "logprobs.jsonl"):
        r = json.loads(l); ana[r["id"]] = {"relations": r["type1"]["C_rho"]["logp"], "source": r["type1"]["C1"]["logp"]}
    bl = {}
    for l in open(Path(reader_cfg["blend_run"]) / "logprobs_blend_block.jsonl"):
        r = json.loads(l); bl[r["id"]] = {"inputs": r["inputs"]["logp"], "generic": r["generic"]["logp"]}
    out = []
    for (test, rid), row in rows.items():
        base = ana.get(rid, {}) if test in ("within_task", "blend_to_analogy") else (bl.get(rid, {}) if test == "analogy_to_blend" else {})
        out.append({"test": test, "id": rid, **base, **row})
    return pd.DataFrame(out)


def analyze(df: pd.DataFrame) -> dict:
    res = {}
    for test, cols in TESTS.items():
        t = df[df.test == test].dropna(subset=cols)
        if len(t) == 0:
            continue
        r = {"n": int(len(t)), "means": {c: float(t[c].mean()) for c in cols}, "gains": {}}
        for a, b in GAINS[test]:
            r["gains"][f"{a}_over_{b}"] = _paired(t[a] - t[b])
        res[test] = r
    return res


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers",):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite); copy_config(config_path, out); (out / "results").mkdir()
    frames, table = {}, {}
    for name, cfg in config["readers"].items():
        if not (Path(cfg["reuse"]) / "logprobs_reuse.jsonl").exists():
            print(f"skip {name}"); continue
        frames[name] = _frame(cfg); table[name] = analyze(frames[name])
    if not frames:
        raise FileNotFoundError("FATAL: no reader has reuse results")
    keys = set.intersection(*(set(zip(f.test, f.id)) for f in frames.values()))
    st = pd.concat([f[[ (a, b) in keys for a, b in zip(f.test, f.id)]] for f in frames.values()])
    num = [c for c in st.columns if c not in ("test", "id")]
    avg = st.groupby(["test", "id"])[num].mean().reset_index()
    table["average_over_readers"] = analyze(avg)
    avg.to_csv(out / "results" / "per_item_reader_average.csv", index=False)
    json.dump(table, open(out / "results" / "reuse.json", "w"), indent=2)
    for test in TESTS:
        print(f"\n=== {test}")
        for name, res in table.items():
            if test not in res:
                continue
            r = res[test]; g = r["gains"]
            print(f"  {name[:22]:22s} n={r['n']:4d} " + " | ".join(f"{k}: {v['mean']:+.1f} ({100*v['frac_positive']:.0f}%)" for k, v in g.items()))
        a = table["average_over_readers"].get(test)
        if a:
            print("  means (avg over readers):", {k: round(v, 1) for k, v in a["means"].items()})
            for k, v in a["gains"].items():
                print(f"  avg {k}: {v['mean']:+.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] {100*v['frac_positive']:.0f}% p={v['wilcoxon_p']:.0e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); main(a.config_path, a.overwrite, a.debug)
