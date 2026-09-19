"""Partition the two information gains by the MODEL that generated the analogy (35 generators).

Per model: within-item means (item means removed, so models are compared on the same anchor
pairs) with bootstrap CIs, raw means, judge pass rates, and the model's KOMBINE analogy composite
and overall score from the upstream composite.json. Correlations across models between the gains
and the KOMBINE scores. Two figures: per-model dot plots; invention gain vs KOMBINE analogy score.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig.scripts.analyze_logprob import type1_frame, type3_frame
from src.analogy_ig.scripts.analyze_family import _boot_ci, family_of

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def _short(model_key: str) -> str:
    return model_key.split("_", 1)[1] if "_" in model_key else model_key


def _style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


def _dots(ax, tab: pd.DataFrame, col: str, title: str, xlabel: str):
    tab = tab.sort_values(f"{col}_mean")
    y = np.arange(len(tab))
    ax.hlines(y, tab[f"{col}_lo"], tab[f"{col}_hi"], color=BLUE, lw=1.6)
    ax.plot(tab[f"{col}_mean"], y, "o", color=BLUE, ms=4.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{_short(m)} ({n})" for m, n in zip(tab.model, tab.n)], fontsize=7.5)
    ax.axvline(0, color=INK2, lw=1, ls=":")
    _style(ax, title, xlabel)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("upstream_dir", "dataset_dir", "composite_json"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out); (out / "results").mkdir(); (out / "figures").mkdir()

    rows = [json.loads(l) for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    comp = json.load(open(config["composite_json"]))["per_model"]

    t1, t3 = type1_frame(rows), type3_frame(rows)
    df = t1.merge(t3[["id", "E_inv", "E_src", "A_B0", "integration", "utility"]], on="id", how="left")
    df = df[df.U_an].copy()
    df["surprise"] = [recs[i]["surprise"] for i in df.id]
    for c in ("G_M_ent_per_tok", "E_inv"):
        df[c + "_within"] = df[c] - df.groupby("prompt_id")[c].transform("mean")

    table = []
    for m, sub in df.groupby("model"):
        row = {"model": m, "family": family_of(m), "n": len(sub)}
        for c in ("G_M_ent_per_tok_within", "E_inv_within", "G_M_ent_per_tok", "E_inv"):
            x = sub[c].dropna().values
            row[f"{c}_mean"] = float(x.mean()); row[f"{c}_lo"], row[f"{c}_hi"] = _boot_ci(x)
        row["integration_rate"] = float(sub.integration.mean())
        row["utility_rate"] = float(sub.utility.mean())
        row["surprise_mean"] = float(sub.surprise.mean())
        c = comp.get(m)
        if c is None:
            raise KeyError(f"FATAL: {m} missing from composite.json")
        row["kombine_analogy"] = c["per_task"]["analogy"]
        row["kombine_overall"] = c["overall"]
        row["kombine_analogy_utility"] = c["raw"]["analogy"]["utility"]
        table.append(row)
    tab = pd.DataFrame(table)
    tab.sort_values("E_inv_within_mean", ascending=False).to_csv(out / "results" / "per_model.csv", index=False)

    def sp(a, b):
        r = stats.spearmanr(tab[a], tab[b]); return {"spearman": float(r.statistic), "p": float(r.pvalue)}
    tests = {
        "invention_gain_vs_kombine_analogy": sp("E_inv_within_mean", "kombine_analogy"),
        "invention_gain_vs_kombine_overall": sp("E_inv_within_mean", "kombine_overall"),
        "invention_gain_vs_integration_rate": sp("E_inv_within_mean", "integration_rate"),
        "alignment_gain_vs_kombine_analogy": sp("G_M_ent_per_tok_within_mean", "kombine_analogy"),
        "alignment_gain_vs_kombine_overall": sp("G_M_ent_per_tok_within_mean", "kombine_overall"),
        "alignment_gain_vs_analogy_utility_rate": sp("G_M_ent_per_tok_within_mean", "kombine_analogy_utility"),
        "alignment_vs_invention_gain_models": sp("G_M_ent_per_tok_within_mean", "E_inv_within_mean"),
        "kruskal_alignment_within": dict(zip(("H", "p"), map(float, stats.kruskal(*[g.G_M_ent_per_tok_within.values for _, g in df.groupby("model") if len(g) >= 5])[:2]))),
        "kruskal_invention_within": dict(zip(("H", "p"), map(float, stats.kruskal(*[g.E_inv_within.dropna().values for _, g in df.groupby("model") if len(g) >= 5])[:2]))),
        "n_models": int(len(tab)),
    }
    with open(out / "results" / "tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    pd.set_option("display.width", 240)
    cols = ["model", "n", "G_M_ent_per_tok_within_mean", "E_inv_within_mean", "integration_rate", "kombine_analogy", "kombine_overall"]
    print(tab.sort_values("E_inv_within_mean", ascending=False)[cols].round(2).to_string(index=False))
    print(json.dumps(tests, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 9.5), dpi=160)
    _dots(axes[0], tab, "G_M_ent_per_tok_within", "Alignment gain by generating model (within item)",
          "alignment gain per entity token, item means removed (nats)")
    _dots(axes[1], tab, "E_inv_within", "Invention gain by generating model (within item)",
          "invention gain, item means removed (nats)")
    fig.tight_layout(); fig.savefig(out / "figures" / "gain_by_model.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160)
    for ax, col, lab, key in ((axes[0], "E_inv_within_mean", "invention gain, item means removed (nats)", "invention_gain_vs_kombine_analogy"),
                              (axes[1], "G_M_ent_per_tok_within_mean", "alignment gain per token, item means removed (nats)", "alignment_gain_vs_kombine_analogy")):
        ax.scatter(tab.kombine_analogy, tab[col], s=22, color=BLUE, alpha=0.8, edgecolor="none")
        for _, r in tab.iterrows():
            ax.annotate(_short(r.model), (r.kombine_analogy, r[col]), fontsize=6, color=INK2,
                        xytext=(3, 2), textcoords="offset points")
        t = tests[key]
        ax.text(0.02, 0.96, f"Spearman {t['spearman']:+.2f}, p = {t['p']:.3f}, n = {len(tab)}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK2)
        _style(ax, "", "KOMBINE analogy composite (% of max)"); ax.set_ylabel(lab, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_title("Invention gain vs. KOMBINE analogy score", loc="left", fontsize=10.5, color=INK)
    axes[1].set_title("Alignment gain vs. KOMBINE analogy score", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout(); fig.savefig(out / "figures" / "gain_vs_kombine.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
