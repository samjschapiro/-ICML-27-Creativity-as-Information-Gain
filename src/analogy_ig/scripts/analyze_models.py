"""Cross-reader sensitivity table: re-run the headline contrasts for every scored reader."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig.scripts.analyze_logprob import type1_frame, type3_frame


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"FATAL: {path}")
    return [json.loads(l) for l in open(path)]


def _delta_p(a, b) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b)
    return float(2 * u.statistic / (len(a) * len(b)) - 1), float(u.pvalue)


def headline(rows: list[dict], recs: dict) -> dict:
    t1, t3 = type1_frame(rows), type3_frame(rows)
    v1 = t1[t1.U_an].dropna(subset=["G_ctrl_ent"])
    v3 = t3[t3.U_an]
    iu = v3[v3.integ_unan]
    out = {"n_records": len(rows), "n_valid": int(t1.U_an.sum())}
    out["alignment_gain_mean"] = float(v1.G_M_ent.mean())
    out["alignment_gain_frac_pos"] = float((v1.G_M_ent > 0).mean())
    out["true_beats_donor_frac"] = float((v1.G_M_ent > v1.G_ctrl_ent).mean())
    rd = t1[t1.reason == "relations_differ"]
    out["align_valid_vs_reldiff_delta"], out["align_valid_vs_reldiff_p"] = _delta_p(
        v1.G_M_ent_per_tok, rd.G_M_ent_per_tok)
    fac = t1[t1.channel == "factual"]
    out["align_valid_vs_factfail_delta"], out["align_valid_vs_factfail_p"] = _delta_p(
        v1.G_M_ent_per_tok, fac.G_M_ent_per_tok)
    out["invention_gain_mean"] = float(v3.E_inv.mean())
    out["known_fact_gain_mean"] = float(v3.E_src.mean())
    out["invention_beats_fact_frac"] = float((v3.E_inv > v3.E_src).mean())
    out["integration_delta"], out["integration_p"] = _delta_p(
        iu[iu.integration == True].E_inv, iu[iu.integration == False].E_inv)
    out["coherence_A_B0_delta_majority"], out["coherence_A_B0_p_majority"] = _delta_p(
        v3[v3.utility == True].A_B0, v3[v3.utility == False].A_B0)
    out["coherence_E_inv_delta_majority"], out["coherence_E_inv_p_majority"] = _delta_p(
        v3[v3.utility == True].E_inv, v3[v3.utility == False].E_inv)
    both = t1.merge(t3, on="id", suffixes=("_t1", "_t3")); both = both[both.U_an_t1]
    out["dissociation_spearman"] = float(stats.spearmanr(both.G_M_ent_per_tok, both.E_inv).statistic)
    sur = np.array([recs[i]["surprise"] for i in v1.id], dtype=float); m = ~np.isnan(sur)
    out["surprise_vs_alignment_spearman"] = float(stats.spearmanr(sur[m], v1.G_M_ent_per_tok.values[m]).statistic)
    sur3 = np.array([recs[i]["surprise"] for i in v3.id], dtype=float); m3 = ~np.isnan(sur3)
    out["surprise_vs_invention_spearman"] = float(stats.spearmanr(sur3[m3], v3.E_inv.values[m3]).statistic)
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
    table = {}
    for name, d in config["readers"].items():
        p = Path(d) / "logprobs.jsonl"
        if not p.exists():
            print(f"skip {name}: {p} missing"); continue
        table[name] = headline(_rows(p), recs)
    df = pd.DataFrame(table).T
    df.to_csv(out / "results" / "readers.csv")
    with open(out / "results" / "readers.json", "w") as f:
        json.dump(table, f, indent=2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
