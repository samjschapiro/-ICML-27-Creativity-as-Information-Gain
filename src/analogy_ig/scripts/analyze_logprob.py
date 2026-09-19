"""Test the type-1 and type-3 predictions (design.md P1a-c, P3a-d, dissociability) on logprobs.jsonl."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"FATAL: {path} not found")
    with open(path) as f:
        return [json.loads(l) for l in f]


def _kind(s: dict, kind: str) -> float:
    return s["logp_by_kind"].get(kind, 0.0)


def type1_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        t = r["type1"]
        if "C0" not in t or "C_rho" not in t or "C1" not in t:
            raise ValueError(f"FATAL: type1 conditions missing for {r['id']}")
        rec = {"id": r["id"], "model": r["model"], "prompt_id": r["prompt_id"], "U_an": r["U_an"],
               "reason": r["pair_structural_reason"], "channel": r["pair_channel"],
               "n_hops": r["n_hops"],
               "G_skel_rel": _kind(t["C_rho"], "relation") - _kind(t["C0"], "relation"),
               "G_M_ent": _kind(t["C1"], "entity") - _kind(t["C_rho"], "entity"),
               "G_M_total": t["C1"]["logp"] - t["C_rho"]["logp"],
               "n_ent_tokens": t["C1"]["n_tokens_by_kind"].get("entity", 0)}
        if "C1_shuf" in t:
            rec["G_ctrl_ent"] = _kind(t["C1_shuf"], "entity") - _kind(t["C_rho"], "entity")
            rec["G_ctrl_total"] = t["C1_shuf"]["logp"] - t["C_rho"]["logp"]
        out.append(rec)
    df = pd.DataFrame(out)
    df["G_M_ent_per_tok"] = df["G_M_ent"] / df["n_ent_tokens"].clip(lower=1)
    return df


def type3_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        t = r.get("type3")
        if not t or "images" not in t:
            continue
        e_inv = [im["B1"]["logp"] - im["B0"]["logp"] for im in t["images"]]
        a_abs = [im["B1"]["logp"] / im["B1"]["n_tokens"] for im in t["images"]]
        a0_abs = [im["B0"]["logp"] / im["B0"]["n_tokens"] for im in t["images"]]
        e_src = [s["S1"]["logp"] - s["S0"]["logp"] for s in t["sources"]]
        a_s0 = [s["S0"]["logp"] / s["S0"]["n_tokens"] for s in t["sources"]]
        out.append({"id": r["id"], "model": r["model"], "prompt_id": r["prompt_id"],
                    "U_an": r["U_an"], "integration": r["integration"], "utility": r["utility"],
                    "integ_unan": r["integration_unanimous"], "util_unan": r["utility_unanimous"],
                    "E_inv": float(np.mean(e_inv)), "A_B1": float(np.mean(a_abs)),
                    "A_B0": float(np.mean(a0_abs)), "E_src": float(np.mean(e_src)),
                    "A_S0": float(np.mean(a_s0)),
                    "n_images": len(e_inv)})
    return pd.DataFrame(out)


def _contrast(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": float(a.mean()) if len(a) else None,
           "mean_b": float(b.mean()) if len(b) else None, "labels": [label_a, label_b]}
    if len(a) > 1 and len(b) > 1:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        res["mannwhitney_p"] = float(u.pvalue)
        res["cliffs_delta"] = float(2 * u.statistic / (len(a) * len(b)) - 1)
    return res


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = {"n": int(len(d)), "mean_diff": float(d.mean()), "frac_positive": float((d > 0).mean())}
    if len(d) > 1:
        res["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    return res


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    if "upstream_dir" not in config:
        raise ValueError("FATAL: 'upstream_dir' required")
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, output_dir)
    (output_dir / "results").mkdir()
    rows = _load(Path(config["upstream_dir"]) / "logprobs.jsonl")

    results = {"n_rows": len(rows)}

    # ---------------- Type 1 ----------------
    t1 = type1_frame(rows)
    t1.to_csv(output_dir / "results" / "type1.csv", index=False)
    valid = t1[t1.U_an]
    results["type1"] = {
        "sanity_G_skel_rel_valid": {"mean": float(valid.G_skel_rel.mean()),
                                    "frac_positive": float((valid.G_skel_rel > 0).mean())},
        "P1a_G_M_ent_valid": {"mean": float(valid.G_M_ent.mean()),
                              "frac_positive": float((valid.G_M_ent > 0).mean()),
                              "wilcoxon_vs_zero_p": float(stats.wilcoxon(valid.G_M_ent).pvalue)
                              if len(valid) > 1 else None},
    }
    if "G_ctrl_ent" in t1.columns:
        vc = valid.dropna(subset=["G_ctrl_ent"])
        results["type1"]["P1a_true_vs_donor_source"] = _paired(vc.G_M_ent, vc.G_ctrl_ent)
    rd = t1[t1.reason == "relations_differ"]
    results["type1"]["P1b_valid_vs_relations_differ"] = _contrast(
        valid.G_M_ent_per_tok, rd.G_M_ent_per_tok, "U_an=1", "relations_differ")
    fac = t1[t1.channel == "factual"]
    results["type1"]["P1b_valid_vs_factual_fail"] = _contrast(
        valid.G_M_ent_per_tok, fac.G_M_ent_per_tok, "U_an=1", "factual_fail")
    if len(valid) > 2:
        rho = stats.spearmanr(valid.n_hops, valid.G_M_ent)
        results["type1"]["P1c_G_M_vs_hops"] = {"spearman": float(rho.statistic), "p": float(rho.pvalue)}
        rho2 = stats.spearmanr(valid.n_hops, valid.G_M_ent_per_tok)
        results["type1"]["P1c_G_M_per_tok_vs_hops"] = {"spearman": float(rho2.statistic), "p": float(rho2.pvalue)}
        results["type1"]["P1a_G_M_ent_per_tok_valid_mean"] = float(valid.G_M_ent_per_tok.mean())

    # ---------------- Type 3 ----------------
    t3 = type3_frame(rows)
    t3.to_csv(output_dir / "results" / "type3.csv", index=False)
    results["type3"] = {"n": int(len(t3))}
    if len(t3):
        v3 = t3[t3.U_an]
        iu = v3[v3.integ_unan]
        uu = v3[v3.util_unan]
        results["type3"]["P3a_E_inv_by_integration"] = _contrast(
            iu[iu.integration == True].E_inv, iu[iu.integration == False].E_inv, "integ=T", "integ=F")
        results["type3"]["P3b_A_B1_by_utility"] = _contrast(
            uu[uu.utility == True].A_B1, uu[uu.utility == False].A_B1, "util=T", "util=F")
        # cross-axis (should be weak): E_inv by utility, A by integration
        results["type3"]["P3c_cross_E_inv_by_utility"] = _contrast(
            uu[uu.utility == True].E_inv, uu[uu.utility == False].E_inv, "util=T", "util=F")
        results["type3"]["P3c_cross_A_B1_by_integration"] = _contrast(
            iu[iu.integration == True].A_B1, iu[iu.integration == False].A_B1, "integ=T", "integ=F")
        # plausibility WITHOUT mapping information (B0) as the cleaner coherence proxy
        results["type3"]["P3b_A_B0_by_utility"] = _contrast(
            uu[uu.utility == True].A_B0, uu[uu.utility == False].A_B0, "util=T", "util=F")
        results["type3"]["P3c_cross_A_B0_by_integration"] = _contrast(
            iu[iu.integration == True].A_B0, iu[iu.integration == False].A_B0, "integ=T", "integ=F")
        # utility=False is rare under unanimity; repeat P3b with MAJORITY verdicts (flagged)
        results["type3"]["P3b_majority_A_B1_by_utility"] = _contrast(
            v3[v3.utility == True].A_B1, v3[v3.utility == False].A_B1, "util=T(maj)", "util=F(maj)")
        results["type3"]["P3b_majority_A_B0_by_utility"] = _contrast(
            v3[v3.utility == True].A_B0, v3[v3.utility == False].A_B0, "util=T(maj)", "util=F(maj)")
        results["type3"]["P3c_majority_E_inv_by_utility"] = _contrast(
            v3[v3.utility == True].E_inv, v3[v3.utility == False].E_inv, "util=T(maj)", "util=F(maj)")
        results["type3"]["P3d_E_inv_vs_E_src"] = _paired(v3.E_inv, v3.E_src)
        # known facts are cheap given their own domain; definitions are expensive given the target
        results["type3"]["P3d_A_S0_vs_A_B0"] = _paired(v3.A_S0, v3.A_B0)
        results["type3"]["corr_E_inv_A_B1"] = float(v3[["E_inv", "A_B1"]].corr().iloc[0, 1]) if len(v3) > 2 else None

    # ---------------- Dissociability ----------------
    both = t1.merge(t3, on="id", suffixes=("_t1", "_t3"))
    both = both[both.U_an_t1]
    if len(both) > 2:
        r = stats.spearmanr(both.G_M_ent_per_tok, both.E_inv)
        results["dissociability_G_M_vs_E_inv"] = {"n": int(len(both)), "spearman": float(r.statistic),
                                                  "p": float(r.pvalue)}

    with open(output_dir / "results" / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
