"""Figures and qualitative examples for the phase-1 report (downstream of score_logprob + dataset)."""

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
from src.analogy_ig.scripts.analyze_logprob import type1_frame, type3_frame

# dataviz reference palette (light mode), fixed slot order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax, title, xlabel, ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _hist2(ax, a, b, la, lb, ca, cb, bins):
    ax.hist(a, bins=bins, histtype="step", lw=2, color=ca, label=la)
    ax.hist(b, bins=bins, histtype="step", lw=2, color=cb, label=lb)
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def fig_type1(t1: pd.DataFrame, out: Path):
    v = t1[t1.U_an].dropna(subset=["G_ctrl_ent"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    bins = np.linspace(-40, 80, 49)
    _hist2(ax, v.G_M_ent.clip(-40, 80), v.G_ctrl_ent.clip(-40, 80),
           "true source path", "donor entities, same relations", BLUE, ORANGE, bins)
    _style(ax, "Alignment gain (553 valid analogies)",
           "nats saved on the target entities vs. knowing only the skeleton", "records")
    fig.tight_layout(); fig.savefig(out / "fig1_alignment_gain.png"); plt.close(fig)


def fig_type3(t3: pd.DataFrame, out: Path):
    v = t3[t3.U_an]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    bins = np.linspace(-30, 80, 45)
    _hist2(ax, v.E_inv.clip(-30, 80), v.E_src.clip(-30, 80),
           "invented concept's triples", "projected concept's triples (known facts)", BLUE, ORANGE, bins)
    _style(ax, "Invention gain vs. known-fact gain (553 valid analogies)",
           "nats saved by seeing the full analogy vs. the triple's own domain", "records")
    fig.tight_layout(); fig.savefig(out / "fig2_invention_gain.png"); plt.close(fig)


def fig_judges(t3: pd.DataFrame, out: Path):
    v = t3[t3.U_an]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=160)
    iu = v[v.integ_unan]
    bins = np.linspace(-20, 80, 41)
    _hist2(axes[0], iu[iu.integration == True].E_inv.clip(-20, 80),
           iu[iu.integration == False].E_inv.clip(-20, 80),
           f"judged faithful (n={int((iu.integration == True).sum())})",
           f"judged unfaithful (n={int((iu.integration == False).sum())})", BLUE, ORANGE, bins)
    _style(axes[0], "Invention gain by integration verdict (unanimous)",
           "invention gain, nats", "records")
    bins2 = np.linspace(-8, 0, 41)
    _hist2(axes[1], v[v.utility == True].A_B0.clip(-8, 0), v[v.utility == False].A_B0.clip(-8, 0),
           f"judged coherent (n={int((v.utility == True).sum())})",
           f"judged incoherent (n={int((v.utility == False).sum())})", BLUE, ORANGE, bins2)
    _style(axes[1], "Plausibility by coherence verdict (majority)",
           "log p per token, target domain only", "records")
    fig.tight_layout(); fig.savefig(out / "fig3_judges.png"); plt.close(fig)


def fig_dissociation(t1: pd.DataFrame, t3: pd.DataFrame, out: Path):
    both = t1.merge(t3, on="id", suffixes=("_t1", "_t3"))
    both = both[both.U_an_t1]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=160)
    ax.scatter(both.G_M_ent_per_tok, both.E_inv, s=14, color=BLUE, alpha=0.55, edgecolor="none")
    _style(ax, "Alignment vs. invention gain (553 valid analogies)",
           "alignment gain per entity token, nats", "invention gain, nats")
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout(); fig.savefig(out / "fig4_dissociation.png"); plt.close(fig)


NAMES = {"C0": "anchor only", "C_rho": "anchor + skeleton", "C1": "true source path",
         "C1_shuf": "donor entities"}


def _fmt_path(p):
    return "\n".join(f"  {h} --{r}--> {t}" for h, r, t in p)


def examples(rows: list[dict], recs: dict, t1: pd.DataFrame, t3: pd.DataFrame, out: Path):
    by_id = {r["id"]: r for r in rows}
    t1 = t1.set_index("id"); t3 = t3.set_index("id")
    picks = {}
    v3 = t3[t3.U_an]
    # A: type-1 showcase: top decile of true-minus-donor gap, then the median of that decile
    v1 = t1[t1.U_an].dropna(subset=["G_ctrl_ent"]).copy()
    # keep the alignment showcase separate from the unfaithful-invention showcase (C): exclude
    # records whose invention the panel unanimously judged unfaithful
    unfaithful = set(t3[(t3.integ_unan) & (t3.integration == False)].index)
    v1 = v1[~v1.index.isin(unfaithful)]
    v1["gap"] = v1.G_M_ent - v1.G_ctrl_ent
    top = v1[v1.gap >= v1.gap.quantile(0.9)].sort_values("gap")
    picks["A_alignment_gain"] = top.index[len(top) // 2]
    # B: faithful invention, unanimous, high E_inv (median of top decile)
    b = v3[(v3.integ_unan) & (v3.integration == True) & (v3.util_unan) & (v3.utility == True)]
    bt = b[b.E_inv >= b.E_inv.quantile(0.9)].sort_values("E_inv")
    picks["B_faithful_invention"] = bt.index[len(bt) // 2]
    # C: unfaithful invention, unanimous, low E_inv
    c = v3[(v3.integ_unan) & (v3.integration == False)]
    ct = c[c.E_inv <= c.E_inv.quantile(0.2)].sort_values("E_inv")
    picks["C_unfaithful_invention"] = ct.index[len(ct) // 2]
    # D: utility=False (majority) with high E_inv and typical plausibility: coherence not seen by code length
    d = v3[(v3.utility == False) & (v3.integration == True)]
    picks["D_faithful_but_incoherent"] = d.sort_values("E_inv").index[-1] if len(d) else None
    # E: factuality failure with a large alignment gain
    e = t1[(t1.channel == "factual")].dropna(subset=["G_ctrl_ent"]).copy()
    e["gap"] = e.G_M_ent - e.G_ctrl_ent
    picks["E_alignment_blind_to_truth"] = e.sort_values("gap").index[-1]

    lines = ["# Qualitative examples (auto-extracted; numbers in nats)\n"]
    dump = {}
    for tag, rid in picks.items():
        if rid is None:
            continue
        row, rec = by_id[rid], recs[rid]
        dump[tag] = {"id": rid, "row": row, "rec": rec}
        lines.append(f"## {tag}: {rec['model']} on {rec['u']} :: {rec['v']} (item {rec['prompt_id']})\n")
        lines.append(f"valid mapping={rec['U_an']} (failure channel={rec['pair_channel']}); panel: faithful={rec['integration']}"
                     f" (unanimous={rec['integration_unanimous']}), coherent={rec['utility']}"
                     f" (unanimous={rec['utility_unanimous']})\n")
        lines.append("Source path:\n```\n" + _fmt_path(rec["path_u"]) + "\n```")
        lines.append("Target path:\n```\n" + _fmt_path(rec["path_v"]) + "\n```")
        t = row["type1"]
        ent = {k: t[k]["logp_by_kind"].get("entity", 0.0) for k in t}
        lines.append("Cost of the target path's entity tokens (log p): " + ", ".join(f"{NAMES[k]}={v:.1f}" for k, v in ent.items()))
        lines.append(f"  alignment gain (true source vs skeleton) = {ent['C1']-ent['C_rho']:.1f}; "
                     f"donor entities vs skeleton = {ent.get('C1_shuf', float('nan'))-ent['C_rho']:.1f}\n")
        t3r = row.get("type3")
        if t3r and "images" in t3r:
            lines.append(f"Projected '{t3r['projected']}' -> invention '{t3r['invention']}'\n")
            lines.append("Projected concept's triples (known facts): gain from the full analogy, and per-token log p given the source domain only:")
            for s in t3r["sources"]:
                h, r, tt = s["triple"]
                lines.append(f"  {h} --{r}--> {tt}: gain={s['S1']['logp']-s['S0']['logp']:+.1f}, "
                             f"plausibility={s['S0']['logp']/s['S0']['n_tokens']:.2f}")
            lines.append("Invented concept's triples: gain from the full analogy, and per-token log p given the target domain only:")
            for im in t3r["images"]:
                h, r, tt = im["triple"]
                lines.append(f"  {h} --{r}--> {tt}: gain={im['B1']['logp']-im['B0']['logp']:+.1f}, "
                             f"plausibility={im['B0']['logp']/im['B0']['n_tokens']:.2f}")
            lines.append("")
            for j in rec["judges"]:
                lines.append(f"  judge {j['model']}: faithful={j['integration']} coherent={j['utility']}")
            lines.append("")
    (out / "examples.md").write_text("\n".join(lines))
    with open(out / "examples.json", "w") as f:
        json.dump(dump, f, indent=1)
    print("\n".join(lines))


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("upstream_dir", "dataset_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    (out / "figures").mkdir()
    rows = [json.loads(l) for l in open(Path(config["upstream_dir"]) / "logprobs.jsonl")]
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    t1, t3 = type1_frame(rows), type3_frame(rows)
    fig_type1(t1, out / "figures"); fig_type3(t3, out / "figures")
    fig_judges(t3, out / "figures"); fig_dissociation(t1, t3, out / "figures")
    examples(rows, recs, t1, t3, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(); main(a.config_path, a.overwrite, a.debug)
