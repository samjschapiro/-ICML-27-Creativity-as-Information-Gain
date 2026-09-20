"""Extract every KOMBINE blend (all models) with judge verdicts into a self-contained JSONL."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.blend_ig.dataset import load_blend_records, record_id


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    if "kombine_dir" not in config:
        raise ValueError("FATAL: 'kombine_dir' is required")
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, output_dir)

    records, dropped = load_blend_records(config["kombine_dir"])
    for r in records:
        r["id"] = record_id(r)
    with open(output_dir / "blends.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    valid = [r for r in records if r["V_bl"]]
    summary = {
        "n_models": len({r["model"] for r in records}),
        "n_records": len(records),
        "dropped": dropped,
        "n_V_bl": len(valid),
        "n_items_in_V_bl": len({r["prompt_id"] for r in valid}),
        "n_unanimous_valid": sum(r["generic_ok_unanimous"] and r["generic_ok"] and r["coherent_unanimous"]
                                 and r["coherent"] and r["double_scope_unanimous"] and r["double_scope"] for r in records),
        "scope_unanimous": Counter(r["scope_unanimous"] for r in records),
        "generic_ok": Counter((r["generic_ok"], r["generic_ok_unanimous"]) for r in records),
        "coherent": Counter((r["coherent"], r["coherent_unanimous"]) for r in records),
        "n_triples": Counter(r["n_triples"] for r in records),
        "tags": Counter(t for r in records for t in r["tags"]),
        "tags_in_V_bl": Counter(t for r in valid for t in r["tags"]),
    }
    summary = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, Counter) else v)
               for k, v in summary.items()}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
