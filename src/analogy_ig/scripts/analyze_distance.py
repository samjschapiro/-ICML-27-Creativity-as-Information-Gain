"""Do alignment gain and invention gain vary with semantic distance?

Two distances: (1) between the two anchor entities (item level, n = 30, from the upstream KOMBINE
anchor_distance analysis: MiniLM cosine distance on labels and on label+description); (2) between
the two paths, i.e. KOMBINE's analogy surprise S_an (record level: mean cosine distance between
aligned entities). Anchor distance is an item property, so its unit is the item (gains averaged
over models within the item); surprise varies by model, so its unit is the record, and a
within-item version (item-demeaned) is reported as well.
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


def _sp(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return {"n": int(m.sum())}
    r = stats.spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "spearman": float(r.statistic), "p": float(r.pvalue)}


def _panel(ax, x, y, title, xlabel, ylabel, note):
    ax.scatter(x, y, s=16, color=BLUE, alpha=0.6, edgecolor="none")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=9, color=INK2)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("upstream_dir", "dataset_dir", "anchor_distance_json"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    (out / "results").mkdir(); (out / "figures").mkdir()

    rows = [json.loads(l) for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    ad = json.load(open(config["anchor_distance_json"]))["distances"]

    t1, t3 = type1_frame(rows), type3_frame(rows)
    df = t1.merge(t3[["id", "E_inv", "E_src", "A_B0"]], on="id", how="left")
    df = df[df.U_an].copy()
    df["surprise"] = [recs[i]["surprise"] for i in df.id]
    df["u"] = [recs[i]["u"] for i in df.id]; df["v"] = [recs[i]["v"] for i in df.id]

    def _dist(u, v, key):
        for k in (f"{u} | {v}", f"{v} | {u}"):
            if k in ad:
                return ad[k][key]
        raise KeyError(f"FATAL: no anchor distance for {u} | {v}")
    df["anchor_label_cos"] = [_dist(u, v, "label_cos") for u, v in zip(df.u, df.v)]
    df["anchor_desc_cos"] = [_dist(u, v, "desc_cos") for u, v in zip(df.u, df.v)]
    df = df.dropna(subset=["surprise"])
    df.to_csv(out / "results" / "records.csv", index=False)

    # item-level frame for anchor distance
    item = df.groupby("prompt_id").agg(
        alignment_gain_per_tok=("G_M_ent_per_tok", "mean"), alignment_gain=("G_M_ent", "mean"),
        invention_gain=("E_inv", "mean"), surprise=("surprise", "mean"),
        anchor_label_cos=("anchor_label_cos", "first"), anchor_desc_cos=("anchor_desc_cos", "first"),
        n=("id", "size")).reset_index()
    item.to_csv(out / "results" / "items.csv", index=False)

    # within-item (demeaned) surprise and gains
    for c in ("surprise", "G_M_ent_per_tok", "E_inv"):
        df[c + "_within"] = df[c] - df.groupby("prompt_id")[c].transform("mean")

    res = {
        "n_records": int(len(df)), "n_items": int(len(item)),
        "surprise_vs_alignment_gain_per_tok": _sp(df.surprise.values, df.G_M_ent_per_tok.values),
        "surprise_vs_alignment_gain_total": _sp(df.surprise.values, df.G_M_ent.values),
        "surprise_vs_invention_gain": _sp(df.surprise.values, df.E_inv.values),
        "surprise_vs_known_fact_gain": _sp(df.surprise.values, df.E_src.values),
        "within_item_surprise_vs_alignment_gain_per_tok": _sp(df.surprise_within.values, df.G_M_ent_per_tok_within.values),
        "within_item_surprise_vs_invention_gain": _sp(df.surprise_within.values, df.E_inv_within.values),
        "anchor_label_cos_vs_alignment_gain_per_tok_item": _sp(item.anchor_label_cos.values, item.alignment_gain_per_tok.values),
        "anchor_label_cos_vs_invention_gain_item": _sp(item.anchor_label_cos.values, item.invention_gain.values),
        "anchor_desc_cos_vs_alignment_gain_per_tok_item": _sp(item.anchor_desc_cos.values, item.alignment_gain_per_tok.values),
        "anchor_desc_cos_vs_invention_gain_item": _sp(item.anchor_desc_cos.values, item.invention_gain.values),
        "anchor_label_cos_vs_surprise_item": _sp(item.anchor_label_cos.values, item.surprise.values),
        "surprise_vs_plausibility_target_only": _sp(df.surprise.values, df.A_B0.values),
    }
    with open(out / "results" / "results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), dpi=160)
    r = res["surprise_vs_alignment_gain_per_tok"]
    _panel(axes[0, 0], df.surprise, df.G_M_ent_per_tok, "Alignment gain vs. path surprise",
           "surprise: mean cosine distance between aligned entities", "alignment gain per entity token, nats",
           f"Spearman {r['spearman']:+.2f}, n = {r['n']}")
    r = res["surprise_vs_invention_gain"]
    _panel(axes[0, 1], df.surprise, df.E_inv, "Invention gain vs. path surprise",
           "surprise: mean cosine distance between aligned entities", "invention gain, nats",
           f"Spearman {r['spearman']:+.2f}, n = {r['n']}")
    r = res["anchor_label_cos_vs_alignment_gain_per_tok_item"]
    _panel(axes[1, 0], item.anchor_label_cos, item.alignment_gain_per_tok, "Alignment gain vs. anchor distance (per item)",
           "cosine distance between the two anchor labels", "mean alignment gain per entity token, nats",
           f"Spearman {r['spearman']:+.2f}, n = {r['n']} items")
    r = res["anchor_label_cos_vs_invention_gain_item"]
    _panel(axes[1, 1], item.anchor_label_cos, item.invention_gain, "Invention gain vs. anchor distance (per item)",
           "cosine distance between the two anchor labels", "mean invention gain, nats",
           f"Spearman {r['spearman']:+.2f}, n = {r['n']} items")
    fig.tight_layout(); fig.savefig(out / "figures" / "gain_vs_distance.png"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); main(a.config_path, a.overwrite, a.debug)
