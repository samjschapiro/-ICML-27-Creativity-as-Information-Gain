"""Estimate type-1 (alignment) and type-3 (concept-invention) information gains with a local LM.

For each analogy record:
  Type 1, target p_v under C0 (v only), C_rho (v + relation skeleton), C1 (v + true source path),
          C1_shuf (v + skeleton with a donor item's source entities).
  Type 3, each image triple under B0 (target domain + bare name) and B1 (full analogy context);
          each source triple under S0 (source domain) and S1 (full context) as the known-fact control.
Writes one JSON line per record with all conditional log-probs; analysis happens downstream.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig import render as R



def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"FATAL: dataset not found: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f]


def _select(records: list[dict], sel: dict) -> list[dict]:
    """Explicit subset selection; every key must be named in config."""
    out = records
    if sel.get("require_U_an"):
        out = [r for r in out if r["U_an"]]
    if sel.get("include_invalid_reasons"):
        keep = set(sel["include_invalid_reasons"])
        out = [r for r in out if r["U_an"] or r["pair_structural_reason"] in keep
               or r["pair_channel"] in keep]
    if sel.get("require_unanimous_panel"):
        out = [r for r in out if r["integration_unanimous"] and r["utility_unanimous"]]
    if sel.get("require_projection"):
        out = [r for r in out if r["projection"] and r["projected"] and r["invention"]]
    return out


def _donor_source(rec: dict, pool: list[dict], rng: random.Random) -> list | None:
    """A source path of the same length from a DIFFERENT item, entities only (relations replaced)."""
    cands = [p for p in pool if p["prompt_id"] != rec["prompt_id"] and p["n_hops"] == rec["n_hops"]]
    if not cands:
        return None
    donor = rng.choice(cands)
    return R.substitute_entities(R.relation_sequence(rec["path_u"]), donor["path_u"])


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    for k in ("upstream_dir", "model", "selection", "seed"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' is required")
    resume = bool(config.get("resume", False))
    out_path = Path(config["output_dir"]) / "logprobs.jsonl"
    done_ids: set[str] = set()
    if resume and out_path.exists():
        output_dir = Path(config["output_dir"])
        with open(out_path) as f:
            done_ids = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"RESUME: {len(done_ids)} records already in {out_path}; appending the rest")
    else:
        output_dir = init_directory(config["output_dir"], overwrite=overwrite)
        copy_config(config_path, output_dir)

    records = _load_records(Path(config["upstream_dir"]) / "analogies.jsonl")
    donor_pool = [r for r in records if r["U_an"]]
    selected = _select(records, config["selection"])
    if done_ids:
        selected = [r for r in selected if r["id"] not in done_ids]
        print(f"RESUME: {len(selected)} records remain")
    if debug:
        n = config.get("debug_limit", 5)
        print(f"DEBUG: scoring first {n} of {len(selected)} selected records")
        selected = selected[:n]
    print(f"records total={len(records)} selected={len(selected)} model={config['model']}")

    rng = random.Random(config["seed"])
    backend = config.get("backend", "mlx")
    if backend == "mlx":
        from src.analogy_ig.logprob import Scorer
        scorer = Scorer(config["model"])
    elif backend == "hf":
        from src.analogy_ig.logprob_hf import ScorerHF
        scorer = ScorerHF(config["model"], dtype=config.get("dtype", "bfloat16"))
    else:
        raise ValueError(f"FATAL: unknown backend '{backend}' (mlx|hf)")
    with open(out_path, "a" if done_ids else "w") as f:
        for rec in tqdm(selected):
            row = {"id": rec["id"], "model": rec["model"], "prompt_id": rec["prompt_id"],
                   "U_an": rec["U_an"], "pair_structural_reason": rec["pair_structural_reason"],
                   "pair_channel": rec["pair_channel"], "n_hops": rec["n_hops"],
                   "integration": rec["integration"], "utility": rec["utility"],
                   "integration_unanimous": rec["integration_unanimous"],
                   "utility_unanimous": rec["utility_unanimous"]}
            u, v, pu, pv, n = rec["u"], rec["v"], rec["path_u"], rec["path_v"], rec["n_hops"]
            # Skeleton for C_rho is the TARGET path's own relations. For U_an = 1 this equals the
            # source's; for relations_differ records it must not be the source's, or the C_rho
            # baseline is a mismatched context and the gain to C1 is inflated.
            rho = R.relation_sequence(pv)

            # ---- Type 1: alignment gain on p_v -------------------------------------------
            target, spans = R.render_path(pv)
            conds = {
                "C0": R.ctx_align(v, None, n, None, None),
                "C_rho": R.ctx_align(v, None, n, rho, None),
                "C1": R.ctx_align(v, u, n, None, pu),
            }
            donor = _donor_source(rec, donor_pool, rng)
            if donor is not None:
                conds["C1_shuf"] = R.ctx_align(v, u, n, None, donor)
            row["type1"] = {}
            for name, msgs in conds.items():
                s = scorer.score(msgs, target, spans)
                s.pop("token_logp")
                row["type1"][name] = s
            row["type1_donor_used"] = donor is not None

            # ---- Type 3: invention gain on image triples; source triples as control ------
            row["type3"] = None
            if rec["projection"] and rec["projected"] and rec["invention"]:
                phi, h = rec["projected"], rec["invention"]
                src_triples = [p["source"] for p in rec["projection"] if p.get("source")]
                img_triples = [p["image"] for p in rec["projection"] if p.get("image")]
                bad = [t for t in src_triples + img_triples if not (isinstance(t, list) and len(t) == 3)]
                if len(src_triples) != len(img_triples) or not img_triples or bad:
                    row["type3"] = {"skipped": "malformed projection", "bad_triples": bad}
                else:
                    # Direction of the projection. Undetectable cases are scored as u -> v and
                    # flagged so analyses can restrict to detected ones.
                    direction = R.projection_direction(pu, pv, src_triples, img_triples)
                    if direction == "v_to_u":
                        s_anchor, t_anchor, s_path, t_path = v, u, pv, pu
                    else:
                        s_anchor, t_anchor, s_path, t_path = u, v, pu, pv
                    row["projection_direction"] = direction if direction != "undetectable" else "assumed_u_to_v"
                    b0 = R.ctx_invention_target_only(t_anchor, t_path, h)
                    b1 = R.ctx_invention_full(s_anchor, t_anchor, s_path, t_path, phi, src_triples, h)
                    s0 = R.ctx_source_fact_only(s_anchor, s_path, phi)
                    images, sources = [], []
                    for t in img_triples:
                        txt, sp = R.render_triple(t)
                        images.append({"triple": t,
                                       "B0": _slim(scorer.score(b0, txt, sp)),
                                       "B1": _slim(scorer.score(b1, txt, sp))})
                    for t in src_triples:
                        txt, sp = R.render_triple(t)
                        s1 = R.ctx_source_fact_full(s_anchor, t_anchor, s_path, t_path, phi, src_triples, h, exclude=t)
                        sources.append({"triple": t,
                                        "S0": _slim(scorer.score(s0, txt, sp)),
                                        "S1": _slim(scorer.score(s1, txt, sp))})
                    row["type3"] = {"projected": phi, "invention": h,
                                    "images": images, "sources": sources}
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"wrote {out_path}")


def _slim(s: dict) -> dict:
    s.pop("token_logp", None)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
