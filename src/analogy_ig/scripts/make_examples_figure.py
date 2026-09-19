"""Concrete-example figure for the paper body: chosen analogies with reader-averaged gains.

For each configured record id: both paths, the projected concept and invention, every invented
triple with its invention gain averaged over the readers, the alignment gain (true source vs
skeleton) and donor control likewise averaged, and the panel verdicts. Emits a LaTeX figure* of
compact boxes (ctxbox environment from setup/configurations.tex) into the paper's media/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config


def esc(t: str) -> str:
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("--", r"\mbox{-}\mbox{-}"), (">", r"\textgreater{}")]:
        t = t.replace(a, b)
    return t


def path_line(path: list) -> str:
    parts = [esc(str(path[0][0]))]
    for h, r, t in path:
        parts.append(f"\\textit{{{esc(str(r))}}} $\\to$ {esc(str(t))}")
    return " ".join(parts)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers", "dataset_dir", "examples", "paper_media_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    recs = {}
    for l in open(Path(config["dataset_dir"]) / "analogies.jsonl"):
        r = json.loads(l); recs[r["id"]] = r
    rows = {}
    for name, d in config["readers"].items():
        rows[name] = {r["id"]: r for r in map(json.loads, open(Path(d) / "logprobs.jsonl"))}
    names = list(rows)
    ctrl = config.get("control", "named")
    if ctrl not in ("format", "named"):
        raise ValueError("FATAL: control must be 'format' or 'named'")
    fmt = {}
    for name, d in config["readers"].items():
        fp = Path(d) / "downstream" / f"{ctrl}_control" / f"logprobs_{ctrl}.jsonl"
        if not fp.exists():
            raise FileNotFoundError(f"FATAL: {ctrl} control missing for {name}: {fp}")
        fmt[name] = {r["id"]: r[f"C_{ctrl}"]["logp"] for r in map(json.loads, open(fp))}

    def avg(fn):
        vals = []
        for n in names:
            try:
                vals.append(fn(rows[n]))
            except (KeyError, TypeError):
                pass
        return float(np.mean(vals)) if vals else float("nan")

    boxes = []
    for ex in config["examples"]:
        rid, title = ex["id"], ex["title"]
        rec = recs[rid]
        t1 = lambda rd, k: rd[rid]["type1"][k]["logp"]  # whole scored line, as in the histograms
        align = avg(lambda rd: t1(rd, "C1") - t1(rd, "C_rho"))
        donor = float(np.mean([fmt[n][rid] - rows[n][rid]["type1"]["C_rho"]["logp"] for n in names]))
        lines = [f"\\textbf{{Source path:}} {path_line(rec['path_u'])}",
                 f"\\textbf{{Target path:}} {path_line(rec['path_v'])}",
                 f"\\textbf{{Target path becomes easier to predict by}} {align:+.1f} nats given the source path ({donor:+.1f} given the control input)"]
        t3 = rows[names[0]][rid].get("type3")
        if t3 and "images" in t3:
            lines.append(f"\\textbf{{Source concept}} `{esc(t3['projected'])}' $\\Rightarrow$ \\textbf{{invented concept}} `{esc(t3['invention'])}'. Facts about the invented concept, and how much easier each becomes to predict given the full analogy:")
            for j, im in enumerate(t3["images"]):
                h, r, t = im["triple"]
                g = avg(lambda rd: rd[rid]["type3"]["images"][j]["B1"]["logp"] - rd[rid]["type3"]["images"][j]["B0"]["logp"])
                lines.append(f"\\quad {esc(str(h))} \\textit{{{esc(str(r))}}} $\\to$ {esc(str(t))}: \\textbf{{{g:+.1f}}}")
            src_g = avg(lambda rd: np.mean([s["S1"]["logp"] - s["S0"]["logp"] for s in rd[rid]["type3"]["sources"]]))
            lines.append(f"\\quad Existing facts about `{esc(t3['projected'])}', scored the same way: {src_g:+.1f} on average")
        verdict = []
        if rec["U_an"]:
            verdict.append("valid mapping")
        else:
            verdict.append(f"invalid mapping ({esc(rec['pair_channel'])})")
        if rec["integration"] is not None:
            verdict.append(f"faithful: {'yes' if rec['integration'] else 'no'}" + (" (unanimous)" if rec["integration_unanimous"] else " (majority)"))
        if rec["utility"] is not None:
            verdict.append(f"coherent: {'yes' if rec['utility'] else 'no'}" + (" (unanimous)" if rec["utility_unanimous"] else " (majority)"))
        lines.append("\\textbf{Judges:} " + "; ".join(verdict))
        model = rec["model"].split("_", 1)[1]
        boxes.append(f"\\begin{{ctxbox}}[{esc(title)} ({esc(model)}, {esc(rec['u'])} :: {esc(rec['v'])})]\n" + "\\\\\n".join(lines) + "\n\\end{ctxbox}")
    caption = config.get("caption", "Concrete examples.")
    tex = ["\\begin{figure*}[t]", "\\centering", "\\scriptsize", "\\begin{minipage}[t]{0.49\\textwidth}\\vspace{0pt}",
           *boxes[: (len(boxes) + 1) // 2], "\\end{minipage}\\hfill", "\\begin{minipage}[t]{0.49\\textwidth}\\vspace{0pt}",
           *boxes[(len(boxes) + 1) // 2:], "\\end{minipage}", f"\\caption{{\\ai{{{caption}}}}}", "\\label{fig:examples}", "\\end{figure*}"]
    text = "\n".join(tex) + "\n"
    (out / "fig_examples.tex").write_text(text)
    Path(config["paper_media_dir"], "fig_examples.tex").write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
