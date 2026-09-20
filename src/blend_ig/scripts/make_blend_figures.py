"""Paper histograms for the blend ladder (Nimbus Roman, ICML column width): log-probability of the
inherited lines (tags u and v) and of the emergent lines under four inputs, one value per blend
averaged over the six readers. Colours follow the alignment figure: input u #D9696B, input v
#F0B75B, both inputs #488AE5, generic space #9CDC9B. No legend, x label or annotation."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils import copy_config, init_directory, load_config

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
COL_W = 3.25
INK, INK2, GRID = "#222222", "#555555", "#e6e6e6"
COLS = (("L_input_u", "#D9696B"), ("L_input_v", "#F0B75B"), ("L_inputs", "#488AE5"), ("L_generic", "#9CDC9B"))
PANELS = (("fig_joint_compression_hist", ["u", "v"]), ("fig_emergent_property_hist", ["emergent"]))


def _hist(ax, F, lo, hi):
    bins = np.linspace(lo, hi, 46)
    for col, colr in COLS:
        x = F[col].clip(lo, hi)
        ax.hist(x, bins=bins, histtype="stepfilled", color=colr, alpha=0.35, lw=0)
        ax.hist(x, bins=bins, histtype="step", color=colr, lw=1.3)
        ax.axvline(x.mean(), color=colr, lw=0.9, ls="--")
    ax.set_xlim(lo, hi); ax.set_xlabel(""); ax.set_ylabel("blends", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(colors=INK2)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    T = pd.read_csv(Path(config["ladder_dir"]) / "results" / "per_triple_reader_average.csv")
    T = T[T.V_bl]
    lo, hi = config.get("x_range", [-60, 0])
    summary = {}
    for name, tags in PANELS:
        F = T[T.tag.isin(tags)].groupby("id")[[c for c, _ in COLS]].sum()
        fig, ax = plt.subplots(figsize=(COL_W, 2.0))
        _hist(ax, F, lo, hi)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        summary[name] = {"n": int(len(F)), **{c: float(F[c].mean()) for c, _ in COLS}}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
