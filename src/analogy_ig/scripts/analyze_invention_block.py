"""Cross-reader ladder for the invention-block measurement.

Scored block: the whole description of the invented concept, M[Phi], as a relational path.
Four inputs, in order of how much of the analogy they reveal:
  structure   task only (skeleton of relations, fact count, the concept's slot)
  instruction task + "use an analogy with <source concept>"
  mapping     task + the two aligned paths with correspondences, no source facts
  content     task + the source concept's facts + the two aligned paths with correspondences
Reports, per reader and averaged over readers, the mean joint log-probability under each input,
the three consecutive gains, the total gain, and paired tests over valid analogies.
Also reports the gain restricted to entity tokens, since relation tokens are given by the task.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config

STEPS = ["structure", "instruction", "mapping", "content"]
FILE = "downstream/invention_block/logprobs_invention_block.jsonl"


def _frame(path: Path, recs: dict) -> pd.DataFrame:
    rows = []
    for l in open(path):
        r = json.loads(l)
        rec = recs[r["id"]]
        row = {"id": r["id"], "model": rec["model"], "U_an": rec["U_an"], "n_facts": r["n_facts"],
               "integration": rec["integration"], "integ_unan": rec["integration_unanimous"],
               "surprise": rec.get("surprise")}
        for s in STEPS:
            row[f"L_{s}"] = r[s]["logp"]
            row[f"Lent_{s}"] = r[s]["logp_by_kind"].get("entity", 0.0)
            row[f"Lrel_{s}"] = r[s]["logp_by_kind"].get("relation", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    out = {"mean": float(d.mean()), "frac_positive": float((d > 0).mean()), "n": int(len(d))}
    if len(d) > 1:
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        out["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    return out


def ladder(df: pd.DataFrame) -> dict:
    v = df[df.U_an]
    out = {"n_records": int(len(df)), "n_valid": int(len(v))}
    for s in STEPS:
        out[f"L_{s}"] = float(v[f"L_{s}"].mean())
    gains = {"instruction_vs_structure": ("instruction", "structure"),
             "mapping_vs_instruction": ("mapping", "instruction"),
             "content_vs_mapping": ("content", "mapping"),
             "mapping_vs_structure": ("mapping", "structure"),
             "content_vs_structure": ("content", "structure")}
    for name, (a, b) in gains.items():
        p = _paired(v[f"L_{a}"] - v[f"L_{b}"])
        out[f"gain_{name}"] = p["mean"]
        out[f"gain_{name}_frac_pos"] = p["frac_positive"]
        out[f"gain_{name}_p"] = p.get("wilcoxon_p")
        out[f"gain_{name}_ci95"] = p.get("ci95")
        pe = _paired(v[f"Lent_{a}"] - v[f"Lent_{b}"])
        out[f"gain_ent_{name}"] = pe["mean"]
    out["L_relation_content"] = float(v["Lrel_content"].mean())
    out["L_relation_structure"] = float(v["Lrel_structure"].mean())
    # share of the total gain that arrives with the mapping alone
    tot = v.L_content - v.L_structure
    out["share_of_total_from_mapping"] = float((v.L_mapping - v.L_structure).sum() / tot.sum())
    # judge visibility: integration pass vs fail among unanimous panels, total gain
    iu = v[v.integ_unan]
    a, b = iu[iu.integration == True], iu[iu.integration == False]
    if len(a) > 1 and len(b) > 1:
        u = stats.mannwhitneyu(tot[a.index], tot[b.index])
        out["integration_cliffs_delta"] = float(2 * u.statistic / (len(a) * len(b)) - 1)
        out["integration_p"] = float(u.pvalue)
    # surprise, within-item (demeaned by generating model x anchor pair is not available here; use raw)
    sur = v.surprise.astype(float); m = ~sur.isna()
    if m.sum() > 10:
        out["surprise_vs_total_spearman"] = float(stats.spearmanr(sur[m], tot[m]).statistic)
    return out


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers", "dataset_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out); (out / "results").mkdir()
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    table, frames = {}, {}
    for name, d in config["readers"].items():
        p = Path(d) / FILE
        if not p.exists():
            print(f"skip {name}: {p} missing"); continue
        frames[name] = _frame(p, recs)
        table[name] = ladder(frames[name])
    if not table:
        raise FileNotFoundError("FATAL: no reader has an invention-block file")
    # cross-reader average: per analogy, average each log-prob over readers, then the same ladder
    common = set.intersection(*(set(f.id) for f in frames.values()))
    stacked = pd.concat([f[f.id.isin(common)].set_index("id") for f in frames.values()])
    num = [c for c in stacked.columns if c.startswith(("L_", "Lent_", "Lrel_"))]
    avg = stacked.groupby(level=0)[num].mean()
    meta = next(iter(frames.values())).set_index("id").loc[avg.index, ["model", "U_an", "n_facts", "integration", "integ_unan", "surprise"]]
    avg = pd.concat([meta, avg], axis=1).reset_index()
    table["average_over_readers"] = ladder(avg)
    avg.to_csv(out / "results" / "per_analogy_reader_average.csv", index=False)
    df = pd.DataFrame(table).T
    df.to_csv(out / "results" / "ladder.csv")
    with open(out / "results" / "ladder.json", "w") as f:
        json.dump(table, f, indent=2)
    show = ["n_valid"] + [f"L_{s}" for s in STEPS] + [
        "gain_instruction_vs_structure", "gain_mapping_vs_instruction", "gain_content_vs_mapping",
        "gain_content_vs_structure", "gain_content_vs_structure_frac_pos", "share_of_total_from_mapping",
        "gain_ent_content_vs_structure", "L_relation_content", "integration_cliffs_delta", "surprise_vs_total_spearman"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df[show].astype(float).round(2).T.to_string())
    print("\nWilcoxon p (content vs structure):")
    print(df["gain_content_vs_structure_p"].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
