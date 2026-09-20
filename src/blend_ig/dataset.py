"""Load KOMBINE blends (concept, generic space, tagged triples) and join them with judge verdicts.

Source: comb-creat-eval/data/kg_creat/kombine_test30/{responses,scores}/<model>/, mode
'blending'. One record per blend. Every field the estimators need is materialised here so
downstream scripts never touch the upstream repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analogy_ig.dataset import _load_json, _unanimous

TAGS = ("u", "v", "uv", "emergent")


def _scope_votes(judges: list[dict]) -> list[int | None]:
    return [j.get("scope") for j in judges]


def load_blend_records(kombine_dir: str | Path) -> tuple[list[dict], dict]:
    """Join responses (artifact) with path_scores (verdicts) for every blend, all models.
    Returns (records, dropped) where dropped counts malformed artifacts by reason."""
    kombine_dir = Path(kombine_dir)
    resp_root, score_root = kombine_dir / "responses", kombine_dir / "scores"
    if not resp_root.is_dir() or not score_root.is_dir():
        raise FileNotFoundError(f"FATAL: expected responses/ and scores/ under {kombine_dir}")
    records, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for model_dir in sorted(score_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("_"):
            continue
        model = model_dir.name
        resp_path = resp_root / model / "responses.json"
        if not resp_path.exists():
            raise FileNotFoundError(f"FATAL: scores exist for {model} but no responses at {resp_path}")
        responses = {(r["prompt_id"], r["temperature"], r["sample_idx"]): r
                     for r in _load_json(resp_path) if r.get("mode") == "blending"}
        scores = [s for s in _load_json(model_dir / "path_scores.json") if s.get("mode") == "blending"]
        for s in scores:
            key = (s["prompt_id"], s["temperature"], s["sample_idx"])
            resp = responses.get(key)
            if resp is None:
                raise KeyError(f"FATAL: no response for {model} {key}")
            items = resp.get("items") or []
            if not items:
                drop("no_artifact"); continue
            item = items[0]
            paths = item.get("paths") or []
            if len(paths) != 1:
                drop("not_one_path"); continue
            triples = paths[0]
            tags = item.get("tags") or []
            if len(tags) != len(triples):
                drop("tags_mismatch"); continue
            if not all(isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) and x for x in t) for t in triples):
                drop("malformed_triple"); continue
            if not (3 <= len(triples) <= 7):
                drop("length"); continue
            concept = item.get("concept")
            generic = item.get("generic_space")
            if not concept or not generic:
                drop("no_concept_or_generic_space"); continue
            judges = s.get("blend_judges") or []
            if len(judges) != 3:
                raise ValueError(f"FATAL: expected a 3-judge panel for {model} {key}, got {len(judges)}")
            generic_maj, generic_unan = _unanimous(judges, "generic_ok")
            coh_maj, coh_unan = _unanimous(judges, "coherent")
            votes = _scope_votes(judges)
            present = [x for x in votes if x is not None]
            double_maj = sum(x >= 2 for x in present) * 2 > len(present) if present else None
            double_unan = (None not in votes) and (all(x >= 2 for x in votes) or all(x < 2 for x in votes))
            scope_unan = votes[0] if (None not in votes and len(set(votes)) == 1) else None
            records.append({
                "model": model,
                "prompt_id": s["prompt_id"],
                "temperature": s["temperature"],
                "sample_idx": s["sample_idx"],
                "u": s["u_label"],
                "v": s["v_label"],
                "domain_u": s.get("domain_u"),
                "domain_v": s.get("domain_v"),
                "concept": concept,
                "generic_space": generic,
                "triples": triples,
                "tags": [t if t in TAGS else "unknown" for t in tags],
                "inferences": item.get("inferences") or [],
                "n_triples": len(triples),
                # KOMBINE blend surprise (R) and originality as scored upstream
                "surprise": s.get("R"),
                "originality": s.get("originality"),
                # panel verdicts: majority and unanimity per dimension; scope votes kept raw
                "generic_ok": generic_maj,
                "generic_ok_unanimous": generic_unan,
                "coherent": coh_maj,
                "coherent_unanimous": coh_unan,
                "scope": s.get("blend_integration"),
                "scope_votes": votes,
                "scope_unanimous": scope_unan,
                "double_scope": double_maj,
                "double_scope_unanimous": double_unan,
                "upstream_generic_ok": s.get("generic_ok"),
                "upstream_blend_utility": s.get("blend_utility"),
                "judges": [{"model": j["model"], "generic_ok": j.get("generic_ok"),
                            "coherent": j.get("coherent"), "scope": j.get("scope")} for j in judges],
            })
            rec = records[-1]
            rec["V_bl"] = bool(rec["generic_ok"] and rec["coherent"] and rec["double_scope"])
    if not records:
        raise ValueError("FATAL: no blend records found")
    return records, dropped


def record_id(rec: dict) -> str:
    return f"{rec['model']}|{rec['prompt_id']}|{rec['temperature']}|{rec['sample_idx']}"
