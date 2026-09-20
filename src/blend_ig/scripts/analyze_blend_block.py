"""Cross-reader analysis of the blend ladder.

Per reader and averaged over readers (valid blends unless stated):
  ladder        mean log-probability of the whole description under each input, the consecutive
                gains (second input, generic space), the vacuous-schema control, the total
  by tag        per-triple gains grouped by the generator's tag: own input, other input, both,
                synergy, generic-space step
  paper gains   joint compression gain (inherited triples, tags u and v) and emergent property
                gain (emergent triples): L(generic) - max(L(input u), L(input v)), summed within
                the blend; reported for the schema with the inputs named and for the schema alone
  judges        generic-space gain by unanimous generic_ok verdict; emergent-triple generic-space
                gain by unanimous scope 3 vs 1; absolute log-probability per token by coherence
  surprise      Spearman of KOMBINE surprise with the total gain
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

INPUTS = ["skeleton", "input_u", "input_v", "inputs", "vacuous", "generic", "generic_only"]
FILE = "logprobs_blend_block.jsonl"


def _triple_logp(cond: dict, i: int) -> float:
    k = cond["logp_by_kind"]
    return k.get(f"{i}:entity", 0.0) + k.get(f"{i}:relation", 0.0)


def _frames(path: Path, recs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks, triples = [], []
    for l in open(path):
        r = json.loads(l)
        rec = recs[r["id"]]
        row = {"id": r["id"], "model": rec["model"], "prompt_id": rec["prompt_id"], "V_bl": rec["V_bl"],
               "n_triples": r["n_triples"], "generic_ok": rec["generic_ok"], "generic_ok_unan": rec["generic_ok_unanimous"],
               "coherent": rec["coherent"], "coherent_unan": rec["coherent_unanimous"],
               "scope_unan": rec["scope_unanimous"], "surprise": rec.get("surprise")}
        for s in INPUTS:
            row[f"L_{s}"] = r[s]["logp"]
        row["n_tokens"] = r["generic"]["n_tokens"]
        blocks.append(row)
        for i, tag in enumerate(r["tags"]):
            t = {"id": r["id"], "V_bl": rec["V_bl"], "tag": tag, "i": i, "scope_unan": rec["scope_unanimous"]}
            for s in INPUTS:
                t[f"L_{s}"] = _triple_logp(r[s], i)
            triples.append(t)
    return pd.DataFrame(blocks), pd.DataFrame(triples)


def _paired(d) -> dict:
    d = np.asarray(d, float)
    out = {"mean": float(d.mean()), "frac_positive": float((d > 0).mean()), "n": int(len(d))}
    if len(d) > 1:
        rng = np.random.default_rng(0)
        boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        out["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return out


def _delta_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b)
    return float(2 * u.statistic / (len(a) * len(b)) - 1), float(u.pvalue)


def analyze(B: pd.DataFrame, T: pd.DataFrame) -> dict:
    v = B[B.V_bl].copy()
    out = {"n_records": int(len(B)), "n_valid": int(len(v))}
    v["L_input_max"] = v[["L_input_u", "L_input_v"]].max(axis=1)
    for s in INPUTS + ["input_max"]:
        out[f"L_{s}"] = float(v[f"L_{s}"].mean())
    gains = {"second_input": ("inputs", "input_max"), "generic_space": ("generic", "inputs"),
             "vacuous_control": ("vacuous", "inputs"), "generic_vs_vacuous": ("generic", "vacuous"),
             "inputs_vs_skeleton": ("inputs", "skeleton"), "total": ("generic", "skeleton")}
    for name, (a, b) in gains.items():
        p = _paired(v[f"L_{a}"] - v[f"L_{b}"])
        out[f"gain_{name}"] = p["mean"]; out[f"gain_{name}_frac_pos"] = p["frac_positive"]
        out[f"gain_{name}_ci95"] = p.get("ci95"); out[f"gain_{name}_p"] = p.get("wilcoxon_p")
    # by tag (triples of valid blends)
    t = T[T.V_bl].copy()
    t["own"] = np.where(t.tag == "v", t.L_input_v - t.L_skeleton, t.L_input_u - t.L_skeleton)
    t["other"] = np.where(t.tag == "v", t.L_input_u - t.L_skeleton, t.L_input_v - t.L_skeleton)
    t["best_single"] = t[["L_input_u", "L_input_v"]].max(axis=1) - t.L_skeleton
    t["both"] = t.L_inputs - t.L_skeleton
    t["synergy"] = t.L_inputs + t.L_skeleton - t.L_input_u - t.L_input_v
    t["generic_step"] = t.L_generic - t.L_inputs
    t["vacuous_step"] = t.L_vacuous - t.L_inputs
    t["generic_only_step"] = t.L_generic_only - t.L_inputs
    by = t.groupby("tag")[["own", "other", "best_single", "both", "synergy", "generic_step", "vacuous_step", "generic_only_step"]].mean()
    by["n"] = t.groupby("tag").size()
    out["by_tag"] = by.round(3).to_dict(orient="index")
    # paper gains (user's framing): the generic space vs the better single input, on the inherited
    # properties (tags u and v; uv reported separately) and on the emergent properties. Summed
    # within the blend, the max taken over the group sums. Two readings of "the generic space":
    # the schema with both inputs named (generic) and the schema alone (generic_only).
    for gcol, label in (("L_generic", "generic"), ("L_generic_only", "generic_only")):
        for tags, name in ((["u", "v"], "joint_compression_gain"), (["uv"], "shared_slot_gain"),
                           (["u", "v", "uv"], "joint_compression_gain_incl_uv"), (["emergent"], "emergent_property_gain")):
            grp = t[t.tag.isin(tags)].groupby("id")[[gcol, "L_input_u", "L_input_v"]].sum()
            if len(grp) == 0:
                continue
            gain = grp[gcol] - grp[["L_input_u", "L_input_v"]].max(axis=1)
            out[f"{name}__{label}"] = _paired(gain)
    # the same groups against the control that names both inputs without the schema: what the
    # generic space itself contributes (figure panel between the single inputs and the schema)
    for tags, name in ((["u", "v"], "joint_compression_gain"), (["uv"], "shared_slot_gain"),
                       (["u", "v", "uv"], "joint_compression_gain_incl_uv"), (["emergent"], "emergent_property_gain")):
        grp = t[t.tag.isin(tags)].groupby("id")[["L_generic", "L_inputs", "L_vacuous"]].sum()
        if len(grp) == 0:
            continue
        out[f"{name}__generic_vs_inputs"] = _paired(grp.L_generic - grp.L_inputs)
        out[f"{name}__vacuous_vs_inputs"] = _paired(grp.L_vacuous - grp.L_inputs)
    # synergy ordering test: emergent + uv vs u + v triples
    a = t[t.tag.isin(["emergent", "uv"])].synergy; b = t[t.tag.isin(["u", "v"])].synergy
    out["synergy_fused_vs_inherited_delta"], out["synergy_fused_vs_inherited_p"] = _delta_p(a, b)
    # judges (all records, both polarities)
    g = B.L_generic - B.L_inputs
    gu = B[B.generic_ok_unan]
    out["generic_gain_by_generic_ok_delta"], out["generic_gain_by_generic_ok_p"] = _delta_p(
        g[gu[gu.generic_ok == True].index], g[gu[gu.generic_ok == False].index])
    out["generic_gain_generic_ok_true_mean"] = float(g[gu[gu.generic_ok == True].index].mean())
    out["generic_gain_generic_ok_false_mean"] = float(g[gu[gu.generic_ok == False].index].mean())
    te = T[T.tag == "emergent"].copy(); te["gs"] = te.L_generic - te.L_inputs
    e3 = te[te.scope_unan == 3].groupby("id").gs.sum(); e1 = te[te.scope_unan == 1].groupby("id").gs.sum()
    out["emergent_generic_gain_scope3_vs_scope1_delta"], out["emergent_generic_gain_scope3_vs_scope1_p"] = _delta_p(e3, e1)
    out["emergent_generic_gain_scope3_mean"], out["emergent_generic_gain_scope1_mean"] = float(e3.mean()) if len(e3) else float("nan"), float(e1.mean()) if len(e1) else float("nan")
    per_tok = B.L_generic / B.n_tokens
    cu = B[B.coherent_unan]
    out["logp_per_token_by_coherent_delta"], out["logp_per_token_by_coherent_p"] = _delta_p(
        per_tok[cu[cu.coherent == True].index], per_tok[cu[cu.coherent == False].index])
    sur = v.surprise.astype(float); m = ~sur.isna()
    if m.sum() > 10:
        out["surprise_vs_total_spearman"] = float(stats.spearmanr(sur[m], (v.L_generic - v.L_skeleton)[m]).statistic)
        out["surprise_vs_generic_gain_spearman"] = float(stats.spearmanr(sur[m], (v.L_generic - v.L_inputs)[m]).statistic)
    return out


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers", "dataset_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out); (out / "results").mkdir()
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "blends.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    table, Bs, Ts = {}, {}, {}
    for name, d in config["readers"].items():
        p = Path(d) / FILE
        if not p.exists():
            print(f"skip {name}: {p} missing"); continue
        Bs[name], Ts[name] = _frames(p, recs)
        table[name] = analyze(Bs[name], Ts[name])
    if not table:
        raise FileNotFoundError("FATAL: no reader has a blend-block file")
    common = set.intersection(*(set(b.id) for b in Bs.values()))
    numB = [f"L_{s}" for s in INPUTS] + ["n_tokens"]
    stB = pd.concat([b[b.id.isin(common)].set_index("id") for b in Bs.values()])
    avgB = stB.groupby(level=0)[numB].mean()
    metaB = next(iter(Bs.values())).set_index("id").loc[avgB.index].drop(columns=numB)
    avgB = pd.concat([metaB, avgB], axis=1).reset_index()
    stT = pd.concat([t[t.id.isin(common)] for t in Ts.values()])
    numT = [f"L_{s}" for s in INPUTS]
    avgT = stT.groupby(["id", "i"])[numT].mean().reset_index()
    metaT = next(iter(Ts.values())).drop(columns=numT)
    avgT = avgT.merge(metaT, on=["id", "i"])
    table["average_over_readers"] = analyze(avgB, avgT)
    avgB.to_csv(out / "results" / "per_blend_reader_average.csv", index=False)
    avgT.to_csv(out / "results" / "per_triple_reader_average.csv", index=False)
    with open(out / "results" / "ladder.json", "w") as f:
        json.dump(table, f, indent=2, default=float)
    flat = {n: {k: v for k, v in r.items() if not isinstance(v, (dict, list))} for n, r in table.items()}
    df = pd.DataFrame(flat).T
    df.to_csv(out / "results" / "ladder.csv")
    show = ["n_valid", "L_skeleton", "L_input_max", "L_inputs", "L_vacuous", "L_generic",
            "gain_second_input", "gain_second_input_frac_pos", "gain_generic_space", "gain_generic_space_frac_pos",
            "gain_vacuous_control", "gain_generic_vs_vacuous_frac_pos", "gain_total", "gain_total_frac_pos",
            "synergy_fused_vs_inherited_delta", "generic_gain_by_generic_ok_delta",
            "emergent_generic_gain_scope3_vs_scope1_delta", "surprise_vs_total_spearman"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df[show].astype(float).round(2).T.to_string())
    print("\nby tag, average over readers (valid blends):")
    print(pd.DataFrame(table["average_over_readers"]["by_tag"]).T.to_string())
    print("\npaper gains, average over readers:")
    for k in sorted(k for k in table["average_over_readers"] if "__" in k):
        print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in table["average_over_readers"][k].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
