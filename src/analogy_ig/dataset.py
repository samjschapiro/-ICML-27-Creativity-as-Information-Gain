"""Load KOMBINE analogy artifacts (paths, mapping, projection) and join them with judge verdicts.

Source: comb-creat-eval/data/kg_creat/kombine_test30/{responses,scores}/<model>/. One record per
analogy HEAD (the even path index of a pair). Every field the estimators need is materialised here
so downstream scripts never touch the upstream repo.
"""

from __future__ import annotations

import json
from pathlib import Path

# Judge-panel keys as written by comb-creat-eval/src/kg_creat/judge.py (wire names on purpose).
_JUDGE_INTEGRATION_KEY = "valid"      # J^qua_an: was the mapping actually used
_JUDGE_UTILITY_KEY = "coherent"       # J^utl_an: is the invention coherent


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"FATAL: missing {path}")
    with open(path) as f:
        return json.load(f)


def _unanimous(judges: list[dict], key: str) -> tuple[bool | None, bool]:
    """(majority verdict, unanimous?) over the per-judge raw verdicts for one dimension."""
    # A judge call that failed upstream leaves a record with only {'model'} (14 of 3111 in
    # kombine_test30). It is an abstention: the majority is over the remaining judges and the
    # panel is never counted as unanimous.
    votes = [j.get(key) for j in judges]
    if any(v is None for v in votes):
        present = [v for v in votes if v is not None]
        if not present:
            return None, False
        return sum(bool(v) for v in present) * 2 > len(present), False
    n_true = sum(bool(v) for v in votes)
    return n_true * 2 > len(votes), (n_true == 0 or n_true == len(votes))


def load_analogy_records(kombine_dir: str | Path) -> list[dict]:
    """Join responses (artifact) with path_scores (verdicts) for every analogy head, all models."""
    kombine_dir = Path(kombine_dir)
    resp_root = kombine_dir / "responses"
    score_root = kombine_dir / "scores"
    if not resp_root.is_dir() or not score_root.is_dir():
        raise FileNotFoundError(f"FATAL: expected responses/ and scores/ under {kombine_dir}")

    records = []
    for model_dir in sorted(score_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        model = model_dir.name
        resp_path = resp_root / model / "responses.json"
        if not resp_path.exists():
            raise FileNotFoundError(f"FATAL: scores exist for {model} but no responses at {resp_path}")
        responses = {(r["prompt_id"], r["temperature"], r["sample_idx"]): r
                     for r in _load_json(resp_path) if r.get("mode") == "analogy"}
        scores = _load_json(model_dir / "path_scores.json")
        heads = [s for s in scores if s.get("mode") == "analogy" and s.get("path_idx", 0) % 2 == 0]
        tails = {(s["prompt_id"], s["temperature"], s["sample_idx"]): s
                 for s in scores if s.get("mode") == "analogy" and s.get("path_idx", 0) % 2 == 1}

        for head in heads:
            key = (head["prompt_id"], head["temperature"], head["sample_idx"])
            resp = responses.get(key)
            if resp is None:
                raise KeyError(f"FATAL: no response for {model} {key}")
            items = resp.get("items") or []
            if not items:
                continue  # parse failure upstream: no artifact to score
            item = items[0]
            paths = item.get("paths") or []
            if len(paths) != 2:
                continue  # malformed artifact: not an analogy pair
            tail = tails.get(key)
            judges = head.get("invention_judges") or []
            if len(judges) != 3:
                raise ValueError(f"FATAL: expected a 3-judge panel for {model} {key}, got {len(judges)}")
            integ_maj, integ_unan = _unanimous(judges, _JUDGE_INTEGRATION_KEY)
            util_maj, util_unan = _unanimous(judges, _JUDGE_UTILITY_KEY)

            records.append({
                "model": model,
                "prompt_id": head["prompt_id"],
                "temperature": head["temperature"],
                "sample_idx": head["sample_idx"],
                "u": head["u_label"],
                "v": head["v_label"],
                "domain_u": head["domain_u"],
                "domain_v": head["domain_v"],
                "path_u": paths[0],
                "path_v": paths[1],
                "n_hops": len(paths[0]),
                "projected": item.get("projected"),
                "invention": item.get("invention"),
                "projection": item.get("projection") or [],
                # utility U_an: structural (relation identity etc.) AND factual (single judge gate)
                # KOMBINE analogy surprise S_an: mean MiniLM cosine distance between aligned entities
                "surprise": head.get("R"),
                "U_an": head.get("pair_sat") is True,
                "pair_channel": head.get("pair_channel"),
                "pair_structural_reason": head.get("pair_structural_reason"),
                "factual_u": head.get("factual"),
                "factual_v": tail.get("factual") if tail else None,
                # emergent-creativity panel (majority) and unanimity per dimension
                "integration": integ_maj,
                "integration_unanimous": integ_unan,
                "utility": util_maj,
                "utility_unanimous": util_unan,
                "judges": [{"model": j["model"], "integration": j.get(_JUDGE_INTEGRATION_KEY),
                            "utility": j.get(_JUDGE_UTILITY_KEY)} for j in judges],
            })
    if not records:
        raise ValueError("FATAL: no analogy records found")
    return records


def record_id(rec: dict) -> str:
    return f"{rec['model']}|{rec['prompt_id']}|{rec['temperature']}|{rec['sample_idx']}"
