"""Partition the two information gains by the model FAMILY that generated the analogy.

Family = provider prefix of the KOMBINE model key (anthropic, openai, google, x-ai, deepseek,
meta-llama, qwen, z-ai, moonshotai, microsoft). Two views:
  raw:        family means over valid analogies (families differ in which items they solved);
  within-item: item means removed first, so a family is compared to the other families on the
              same anchor pairs (removes item difficulty as a confound).
Per-model means are written too. Tests: Kruskal-Wallis across families; bootstrap CIs on means.
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

BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
FAMILY_LABEL = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "x-ai": "xAI",
                "deepseek": "DeepSeek", "meta-llama": "Meta", "qwen": "Qwen", "z-ai": "Zhipu",
                "moonshotai": "Moonshot", "microsoft": "Microsoft"}


def family_of(model_key: str) -> str:
    return model_key.split("_", 1)[0]


def _boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float("nan"), float("nan")
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summary(df: pd.DataFrame, group: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group):
        row = {group: g, "n": len(sub)}
        for c in cols:
            x = sub[c].dropna().values
            lo, hi = _boot_ci(x)
            row[f"{c}_mean"] = float(x.mean()) if len(x) else float("nan")
            row[f"{c}_lo"], row[f"{c}_hi"] = lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def _dotplot(ax, summ: pd.DataFrame, col: str, label: str, title: str):
    summ = summ.sort_values(f"{col}_mean")
    y = np.arange(len(summ))
    ax.hlines(y, summ[f"{col}_lo"], summ[f"{col}_hi"], color=BLUE, lw=2)
    ax.plot(summ[f"{col}_mean"], y, "o", color=BLUE, ms=6)
    ax.set_yticks(y); ax.set_yticklabels([f"{FAMILY_LABEL.get(f, f)} (n={n})" for f, n in zip(summ.family, summ.n)])
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(label, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("upstream_dir", "dataset_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out); (out / "results").mkdir(); (out / "figures").mkdir()

    rows = [json.loads(l) for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    t1, t3 = type1_frame(rows), type3_frame(rows)
    df = t1.merge(t3[["id", "E_inv", "E_src", "A_B0"]], on="id", how="left")
    df = df[df.U_an].copy()
    df["family"] = df.model.map(family_of)
    df["surprise"] = [recs[i]["surprise"] for i in df.id]
    cols = ["G_M_ent_per_tok", "E_inv", "E_src", "surprise"]
    for c in cols:
        df[c + "_within"] = df[c] - df.groupby("prompt_id")[c].transform("mean")

    fam_raw = _summary(df, "family", cols)
    fam_within = _summary(df, "family", [c + "_within" for c in cols])
    per_model = _summary(df, "model", cols)
    per_model["family"] = per_model.model.map(family_of)
    fam_raw.to_csv(out / "results" / "family_raw.csv", index=False)
    fam_within.to_csv(out / "results" / "family_within_item.csv", index=False)
    per_model.sort_values("G_M_ent_per_tok_mean").to_csv(out / "results" / "per_model.csv", index=False)

    # validity rate by family (share of that family's analogies with U_an = 1), from the full dataset
    all_df = pd.DataFrame([{"model": r["model"], "U_an": r["U_an"]} for r in recs.values()])
    all_df["family"] = all_df.model.map(family_of)
    valid_rate = all_df.groupby("family").U_an.mean().rename("valid_rate")

    tests = {}
    for c in cols + [c + "_within" for c in cols]:
        groups = [g[c].dropna().values for _, g in df.groupby("family") if len(g) >= 5]
        kw = stats.kruskal(*groups)
        tests[c] = {"kruskal_H": float(kw.statistic), "p": float(kw.pvalue), "n_groups": len(groups)}
    # does a family's validity rate track its gains?
    fr = fam_raw.set_index("family").join(valid_rate)
    tests["valid_rate_vs_alignment_gain_family"] = dict(zip(("spearman", "p"), map(float, stats.spearmanr(fr.valid_rate, fr.G_M_ent_per_tok_mean)[:2])))
    tests["valid_rate_vs_invention_gain_family"] = dict(zip(("spearman", "p"), map(float, stats.spearmanr(fr.valid_rate, fr.E_inv_mean)[:2])))
    with open(out / "results" / "tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    pd.set_option("display.width", 220)
    print("== family, raw (valid analogies)")
    print(fam_raw.join(valid_rate, on="family").round(3).to_string(index=False))
    print("\n== family, within item (item means removed)")
    print(fam_within.round(3).to_string(index=False))
    print("\n== tests"); print(json.dumps(tests, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)
    _dotplot(axes[0], fam_within, "G_M_ent_per_tok_within", "alignment gain per entity token, item means removed (nats)",
             "Alignment gain by generating family (within item)")
    _dotplot(axes[1], fam_within, "E_inv_within", "invention gain, item means removed (nats)",
             "Invention gain by generating family (within item)")
    fig.tight_layout(); fig.savefig(out / "figures" / "gain_by_family.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
