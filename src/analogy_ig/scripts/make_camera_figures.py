"""Camera-ready figures (ICML: Nimbus Roman, column/text widths, PDF) across all readers.

Each headline quantity is shown per reader with a 95% bootstrap CI over analogies, plus an
"Average" row: the mean of the per-reader values with a t-based 95% CI across readers. Per-model
and per-family panels show the reader-averaged within-item gain with the individual readers as
faint points. Absolute nats differ by reader; the within-reader contrasts are what generalise.
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
from src.analogy_ig.scripts.analyze_family import family_of, FAMILY_LABEL

# ICML geometry and type
COL_W, TEXT_W = 3.25, 6.75
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, AVG = "#0b0b0b", "#52514e", "#dddcd7", "#0b0b0b"
plt.rcParams.update({
    "font.family": "Nimbus Roman", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


CONTROL = "named"


def _load(readers: dict, dataset_dir: Path, composite_json: Path) -> dict:
    recs = {}
    for l in open(dataset_dir / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    comp = json.load(open(composite_json))["per_model"]
    out = {}
    for name, d in readers.items():
        rows = [json.loads(l) for l in open(Path(d) / "logprobs.jsonl")]
        t1, t3 = type1_frame(rows), type3_frame(rows)
        df = t1.merge(t3[["id", "E_inv", "E_src", "A_B0", "integration", "utility", "integ_unan", "util_unan"]],
                      on="id", how="left")
        df = df[df.U_an].copy()
        df["family"] = df.model.map(family_of)
        df["surprise"] = [recs[i]["surprise"] for i in df.id]
        for c in ("G_M_ent_per_tok", "E_inv", "surprise"):
            df[c + "_w"] = df[c] - df.groupby("prompt_id")[c].transform("mean")
        # format control (content-free source path), scored as a downstream pass of the run
        ctrl = CONTROL
        fpath = Path(d) / "downstream" / f"{ctrl}_control" / f"logprobs_{ctrl}.jsonl"
        if not fpath.exists():
            raise FileNotFoundError(f"FATAL: {ctrl} control missing for {name}: {fpath}")
        fmt = {r["id"]: r[f"C_{ctrl}"]["logp"] for r in map(json.loads, open(fpath))}
        base = {r["id"]: r["type1"]["C_rho"]["logp"] for r in rows}
        real = {r["id"]: r["type1"]["C1"]["logp"] for r in rows}
        df["G_fmt_total"] = [fmt[i] - base[i] for i in df.id]
        df["L_structure"] = [base[i] for i in df.id]   # relations only
        df["L_instruction"] = [fmt[i] for i in df.id]  # control (named source)
        df["L_content"] = [real[i] for i in df.id]     # real source path
        out[name] = df
    return out, comp


def _boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def _cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def _boot_cliff(a, b, n=1000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    d = np.array([_cliff(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)])
    return _cliff(a, b), np.percentile(d, 2.5), np.percentile(d, 97.5)


def _boot_spearman(x, y, n=1000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    r = np.array([stats.spearmanr(x[i], y[i]).statistic for i in (rng.choice(idx, len(idx)) for _ in range(n))])
    return stats.spearmanr(x, y).statistic, np.percentile(r, 2.5), np.percentile(r, 97.5)


def _avg_row(vals):
    vals = np.asarray(vals, float)
    m = vals.mean(); h = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    return m, m - h, m + h


def _style(ax, xlabel, title=None):
    ax.set_xlabel(xlabel, color=INK)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)


def _reader_rows(ax, labels, series, xlabel, title=None, zero=True):
    """series: list of (name, colour, [(mean, lo, hi) per reader]); an Average row is appended."""
    n_r = len(labels); k = len(series)
    y = np.arange(n_r + 1)[::-1]  # top row = first reader, bottom = Average
    off = np.linspace(-0.22, 0.22, k) if k > 1 else [0.0]
    for (name, col, vals), o in zip(series, off):
        means = [v[0] for v in vals]; los = [v[1] for v in vals]; his = [v[2] for v in vals]
        am, alo, ahi = _avg_row(means)
        yy = y[:n_r] + o
        ax.hlines(yy, los, his, color=col, lw=1.2)
        ax.plot(means, yy, "o", color=col, ms=3.6, label=name, mec="white", mew=0.4)
        ax.hlines(y[n_r] + o, alo, ahi, color=col, lw=1.6)
        ax.plot([am], [y[n_r] + o], "D", color=col, ms=4.2, mec=AVG, mew=0.6)
    if zero:
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.axhline(y[n_r] + 0.5, color=GRID, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(list(labels) + ["Average"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    _style(ax, xlabel, title)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers", "reader_labels", "dataset_dir", "composite_json"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    global CONTROL
    CONTROL = config.get("control", "named")
    if CONTROL not in ("format", "named"):
        raise ValueError("FATAL: control must be 'format' or 'named'")
    data, comp = _load(config["readers"], Path(config["dataset_dir"]), Path(config["composite_json"]))
    names = list(config["readers"]); labels = [config["reader_labels"][n] for n in names]
    summary = {}

    def save(fig, stem):
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out / f"{stem}.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # ---- Fig 1: alignment gain, true source vs donor, per reader + average ----------------
    true_v = [_boot_mean(data[n].G_M_ent) for n in names]
    donor_v = [_boot_mean(data[n].G_ctrl_ent) for n in names]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _reader_rows(ax, labels, [("true source path", BLUE, true_v), ("donor entities (control)", ORANGE, donor_v)],
                 "alignment gain on target entities (nats)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    save(fig, "fig_alignment_gain_readers")
    summary["alignment"] = {n: {"true": true_v[i][0], "donor": donor_v[i][0]} for i, n in enumerate(names)}

    # ---- Fig 2: invention gain vs known-fact gain ------------------------------------------
    inv_v = [_boot_mean(data[n].E_inv) for n in names]
    fact_v = [_boot_mean(data[n].E_src) for n in names]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _reader_rows(ax, labels, [("invented concept's triples", BLUE, inv_v), ("projected concept's known facts", ORANGE, fact_v)],
                 "gain from the full analogy (nats)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    save(fig, "fig_invention_gain_readers")
    summary["invention"] = {n: {"invention": inv_v[i][0], "fact": fact_v[i][0]} for i, n in enumerate(names)}

    # ---- Fig 1b: raw log-probability of the target path under three inputs, reader-averaged ---
    RED, BLU, GRN = "#D9696B", "#488AE5", "#9CDC9B"
    series = {}
    for col in ("L_structure", "L_instruction", "L_content"):
        series[col] = pd.concat([data[n].set_index("id")[col].rename(n) for n in names], axis=1).mean(axis=1)
    L = pd.DataFrame(series).dropna()
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    lo, hi = -110, 0
    bins = np.linspace(lo, hi, 46)
    for col, colr in (("L_structure", RED), ("L_instruction", BLU), ("L_content", GRN)):
        x = L[col].clip(lo, hi)
        ax.hist(x, bins=bins, histtype="stepfilled", color=colr, alpha=0.35, lw=0)
        ax.hist(x, bins=bins, histtype="step", color=colr, lw=1.3)
        ax.axvline(x.mean(), color=colr, lw=0.9, ls="--")
    ax.set_xlim(lo, hi); ax.set_xlabel(""); ax.set_ylabel("analogies", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(colors=INK2)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    save(fig, "fig_alignment_gain_hist")
    summary["alignment_hist"] = {"n": int(len(L)), "mean_structure": float(L.L_structure.mean()),
                                 "mean_instruction": float(L.L_instruction.mean()), "mean_content": float(L.L_content.mean()),
                                 "colors": {"relational structure": RED, "analogy instruction": BLU, "analogy content": GRN}}

    # ---- Fig 2b: histogram of per-analogy gains averaged over the six readers --------------
    inv = pd.concat([data[n].set_index("id").E_inv.rename(n) for n in names], axis=1).mean(axis=1)
    fact = pd.concat([data[n].set_index("id").E_src.rename(n) for n in names], axis=1).mean(axis=1)
    both = pd.concat([inv.rename("inv"), fact.rename("fact")], axis=1).dropna()
    frac = float((both.inv > both.fact).mean())
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    lo, hi = -30, 50
    bins = np.linspace(lo, hi, 46)
    ax.hist(both.inv.clip(lo, hi), bins=bins, histtype="stepfilled", color=BLUE, alpha=0.25, lw=0)
    ax.hist(both.inv.clip(lo, hi), bins=bins, histtype="step", color=BLUE, lw=1.2, label="facts about the invented concept")
    ax.hist(both.fact.clip(lo, hi), bins=bins, histtype="stepfilled", color=ORANGE, alpha=0.25, lw=0)
    ax.hist(both.fact.clip(lo, hi), bins=bins, histtype="step", color=ORANGE, lw=1.2, label="existing facts about the source concept")
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.axvline(both.inv.mean(), color=BLUE, lw=0.8, ls="--"); ax.axvline(both.fact.mean(), color=ORANGE, lw=0.8, ls="--")
    ax.text(0.98, 0.95, f"invented concept higher\nfor {frac:.1%} of analogies", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    ax.set_ylabel("analogies", color=INK); ax.set_xlim(lo, hi)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(colors=INK2)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.set_xlabel("increase in log-probability of the fact (nats), mean over six language models", color=INK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    save(fig, "fig_invention_gain_hist")
    summary["invention_hist"] = {"n": int(len(both)), "mean_invention": float(both.inv.mean()),
                                 "mean_fact": float(both.fact.mean()), "frac_invention_gt_fact": frac}

    # ---- Fig 3: judge contrasts (Cliff's delta) --------------------------------------------
    integ_v, coh_v = [], []
    for n in names:
        df = data[n]; iu = df[df.integ_unan == True]
        integ_v.append(_boot_cliff(iu[iu.integration == True].E_inv, iu[iu.integration == False].E_inv))
        coh_v.append(_boot_cliff(df[df.utility == True].A_B0, df[df.utility == False].A_B0))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _reader_rows(ax, labels, [("integration judge: invention gain", BLUE, integ_v),
                              ("coherence judge: plausibility", ORANGE, coh_v)], "Cliff's delta (judged True minus False)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    save(fig, "fig_judges_readers")
    summary["judges"] = {n: {"integration_delta": integ_v[i][0], "coherence_delta": coh_v[i][0]} for i, n in enumerate(names)}

    # ---- Fig 4: correlations (within item) + dissociation ----------------------------------
    sa_v, si_v, ds_v = [], [], []
    for n in names:
        df = data[n]
        sa_v.append(_boot_spearman(df.surprise_w, df.G_M_ent_per_tok_w))
        si_v.append(_boot_spearman(df.surprise_w, df.E_inv_w))
        ds_v.append(_boot_spearman(df.G_M_ent_per_tok, df.E_inv))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    _reader_rows(ax, labels, [("surprise vs. alignment gain", BLUE, sa_v), ("surprise vs. invention gain", ORANGE, si_v),
                              ("alignment vs. invention gain", AQUA, ds_v)], "Spearman correlation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    save(fig, "fig_correlations_readers")
    summary["correlations"] = {n: {"surprise_alignment": sa_v[i][0], "surprise_invention": si_v[i][0],
                                   "dissociation": ds_v[i][0]} for i, n in enumerate(names)}

    # ---- Fig 5: per-model gains, reader-averaged (text width) ------------------------------
    pm = {n: data[n].groupby("model")[["G_M_ent_per_tok_w", "E_inv_w"]].mean() for n in names}
    models = sorted(set.intersection(*[set(pm[n].index) for n in names]))
    counts = data[names[0]].groupby("model").size()
    tab = pd.DataFrame({"model": models})
    for c in ("G_M_ent_per_tok_w", "E_inv_w"):
        M = np.array([[pm[n].loc[m, c] for n in names] for m in models])
        tab[c + "_avg"] = M.mean(1); tab[c + "_min"] = M.min(1); tab[c + "_max"] = M.max(1)
        for i, n in enumerate(names):
            tab[f"{c}_{n}"] = M[:, i]
    tab["n"] = [int(counts[m]) for m in models]
    tab.to_csv(out / "per_model_reader_average.csv", index=False)

    def dots(ax, col, xlabel, title):
        t = tab.sort_values(col + "_avg"); y = np.arange(len(t))
        for n in names:
            ax.plot(t[f"{col}_{n}"], y, "o", color=BLUE, alpha=0.22, ms=2.6, mec="none")
        ax.hlines(y, t[col + "_min"], t[col + "_max"], color=BLUE, alpha=0.25, lw=0.8)
        ax.plot(t[col + "_avg"], y, "D", color=INK, ms=3.4, label="average over six readers")
        ax.plot([], [], "o", color=BLUE, alpha=0.35, ms=2.6, mec="none", label="individual reader")
        ax.set_yticks(y); ax.set_yticklabels([f"{m.split('_', 1)[1]} ({k})" for m, k in zip(t.model, t.n)], fontsize=6.2)
        ax.axvline(0, color=INK2, lw=0.6, ls=":"); _style(ax, xlabel, title); ax.grid(axis="x", color=GRID, lw=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 5.6))
    dots(axes[0], "G_M_ent_per_tok_w", "alignment gain per entity token, item means removed (nats)", "Alignment gain by generating model")
    dots(axes[1], "E_inv_w", "invention gain, item means removed (nats)", "Invention gain by generating model")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.005), handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.025, 1, 1))
    save(fig, "fig_gain_by_model_readers")

    # ---- Fig 6: per-family gains, reader-averaged (column width) ---------------------------
    pf = {n: data[n].groupby("family")[["G_M_ent_per_tok_w", "E_inv_w"]].mean() for n in names}
    fams = sorted(set.intersection(*[set(pf[n].index) for n in names]))
    fcount = data[names[0]].groupby("family").size()
    ftab = pd.DataFrame({"family": fams})
    for c in ("G_M_ent_per_tok_w", "E_inv_w"):
        M = np.array([[pf[n].loc[f, c] for n in names] for f in fams])
        ftab[c + "_avg"] = M.mean(1); ftab[c + "_min"] = M.min(1); ftab[c + "_max"] = M.max(1)
        for i, n in enumerate(names):
            ftab[f"{c}_{n}"] = M[:, i]
    ftab["n"] = [int(fcount[f]) for f in fams]
    ftab.to_csv(out / "per_family_reader_average.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4))
    for ax, c, xl, ti in ((axes[0], "G_M_ent_per_tok_w", "alignment gain per entity token, item means removed (nats)", "Alignment gain by family"),
                          (axes[1], "E_inv_w", "invention gain, item means removed (nats)", "Invention gain by family")):
        t = ftab.sort_values(c + "_avg"); y = np.arange(len(t))
        for n in names:
            ax.plot(t[f"{c}_{n}"], y, "o", color=BLUE, alpha=0.22, ms=3, mec="none")
        ax.hlines(y, t[c + "_min"], t[c + "_max"], color=BLUE, alpha=0.25, lw=0.9)
        ax.plot(t[c + "_avg"], y, "D", color=INK, ms=3.6, label="average over six readers")
        ax.plot([], [], "o", color=BLUE, alpha=0.35, ms=3, mec="none", label="individual reader")
        ax.set_yticks(y); ax.set_yticklabels([f"{FAMILY_LABEL.get(f, f)} ({k})" for f, k in zip(t.family, t.n)])
        ax.axvline(0, color=INK2, lw=0.6, ls=":"); _style(ax, xl, ti)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.06, 1, 1))
    save(fig, "fig_gain_by_family_readers")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    def _round(o):
        if isinstance(o, dict):
            return {k: _round(v) for k, v in o.items()}
        return round(o, 3) if isinstance(o, float) else o
    print(json.dumps(_round(summary), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
