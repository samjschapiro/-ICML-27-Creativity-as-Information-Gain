"""Extract every KOMBINE analogy head (all models) into a self-contained JSONL for this track."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig.dataset import load_analogy_records, record_id


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    if "kombine_dir" not in config:
        raise ValueError("FATAL: 'kombine_dir' is required")
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, output_dir)

    records = load_analogy_records(config["kombine_dir"])
    for r in records:
        r["id"] = record_id(r)

    with open(output_dir / "analogies.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    strict = [r for r in records if r["U_an"] and r["integration_unanimous"] and r["utility_unanimous"]
              and r["integration"] and r["utility"]]
    summary = {
        "n_models": len({r["model"] for r in records}),
        "n_records": len(records),
        "n_U_an": sum(r["U_an"] for r in records),
        "n_strict_unanimous_true": len(strict),
        "n_items_in_strict": len({r["prompt_id"] for r in strict}),
        "pair_channels": Counter((r["pair_channel"], r["pair_structural_reason"]) for r in records),
        "hops": Counter(r["n_hops"] for r in records),
        "integration_unanimous": Counter((r["integration"], r["integration_unanimous"]) for r in records),
        "utility_unanimous": Counter((r["utility"], r["utility_unanimous"]) for r in records),
    }
    summary = {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, Counter) else v)
               for k, v in summary.items()}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
