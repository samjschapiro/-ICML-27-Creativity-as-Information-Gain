"""Main-text figure: one analogy, the input without and with the analogy, and the log-probability
a language model assigns to the same scored line under each. Two rows: the target path, then one
fact about the invented concept. Log-probabilities are the mean over the configured readers.
Emits LaTeX (two ctxbox columns per row) into the paper's media/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils import copy_config, init_directory, load_config
from src.analogy_ig import render as R
from src.analogy_ig.scripts.make_examples_figure import esc


def latexify(text: str) -> str:
    return "\\\\\n".join(esc(line) if line else "~" for line in text.split("\n"))


def box(title, prompt, scored, logp):
    return (f"\\begin{{ctxbox}}[{title}]\n{latexify(prompt)}\\\\\n~\\\\\n"
            f"\\textbf{{Scored line:}} {latexify(scored)}\\\\\n"
            f"\\textbf{{Log-probability of the scored line:}} {logp:.1f} nats\n\\end{{ctxbox}}")


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for k in ("readers", "dataset_dir", "record_id", "paper_media_dir"):
        if k not in config:
            raise ValueError(f"FATAL: '{k}' required")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    copy_config(config_path, out)
    rid = config["record_id"]
    rec = next(json.loads(l) for l in open(Path(config["dataset_dir"]) / "analogies.jsonl") if json.loads(l)["id"] == rid)
    rows = {}
    for name, d in config["readers"].items():
        rows[name] = next(r for r in map(json.loads, open(Path(d) / "logprobs.jsonl")) if r["id"] == rid)
    names = list(rows)
    mean = lambda fn: float(np.mean([fn(rows[n]) for n in names]))

    u, v, pu, pv, n = rec["u"], rec["v"], rec["path_u"], rec["path_v"], rec["n_hops"]
    rho = R.relation_sequence(pv)
    target, _ = R.render_path(pv)
    lp_rel = mean(lambda r: r["type1"]["C_rho"]["logp"]); lp_src = mean(lambda r: r["type1"]["C1"]["logp"])
    ctrl = config.get("control", "named")
    if ctrl not in ("format", "named"):
        raise ValueError("FATAL: control must be 'format' or 'named'")
    fpath = lambda d: Path(d) / "downstream" / f"{ctrl}_control" / f"logprobs_{ctrl}.jsonl"
    fmt = {}
    for name, d in config["readers"].items():
        if not fpath(d).exists():
            raise FileNotFoundError(f"FATAL: {ctrl} control missing for {name}: {fpath(d)}")
        fmt[name] = next(r for r in map(json.loads, open(fpath(d))) if r["id"] == rid)[f"C_{ctrl}"]["logp"]
    lp_fmt = float(np.mean(list(fmt.values())))
    ctrl_title = {"format": "Control input: the same format with a content-free path",
                  "named": "Control input: the source concept is named, its path is not shown"}[ctrl]
    ctrl_prompt = (R.ctx_align_format(v, n, rho) if ctrl == "format" else R.ctx_align_named(v, u, n, rho))[1]["content"]
    row1 = [box("Baseline input: the target concept and the relations of its path",
                R.ctx_align(v, None, n, rho, None)[1]["content"], target, lp_rel),
            box(ctrl_title, ctrl_prompt, target, lp_fmt),
            box("Analogy input: the real source path",
                R.ctx_align(v, u, n, None, pu)[1]["content"], target, lp_src)]

    t3 = rows[names[0]]["type3"]
    src = [p["source"] for p in rec["projection"]]; img = [p["image"] for p in rec["projection"]]
    d = R.projection_direction(pu, pv, src, img)
    sa, ta, sp, tp = (v, u, pv, pu) if d == "v_to_u" else (u, v, pu, pv)
    gains = [mean(lambda r, j=j: r["type3"]["images"][j]["B1"]["logp"] - r["type3"]["images"][j]["B0"]["logp"]) for j in range(len(t3["images"]))]
    j = int(np.argmax(gains)) if config.get("fact_index") is None else int(config["fact_index"])
    fact, _ = R.render_triple(t3["images"][j]["triple"])
    lp_b0 = mean(lambda r: r["type3"]["images"][j]["B0"]["logp"]); lp_b1 = mean(lambda r: r["type3"]["images"][j]["B1"]["logp"])
    # control for row (b): an existing fact about the source concept, scored under the same two inputs
    k = j if j < len(t3["sources"]) else 0
    sfact, _ = R.render_triple(t3["sources"][k]["triple"])
    lp_s0 = mean(lambda r: r["type3"]["sources"][k]["S0"]["logp"]); lp_s1 = mean(lambda r: r["type3"]["sources"][k]["S1"]["logp"])
    ctrl = (f"\\begin{{ctxbox}}[Control: an existing fact about the source concept, scored the same way]\n"
            f"The fact `{esc(sfact)}' is already true of `{esc(t3['projected'])}'. It is scored after an input giving only the "
            f"source domain and the concept's name, and after the analogy input at right with this fact removed from the list.\\\\\n~\\\\\n"
            f"\\textbf{{Scored line:}} {latexify(sfact)}\\\\\n"
            f"\\textbf{{Log-probability without the analogy:}} {lp_s0:.1f} nats\\\\\n"
            f"\\textbf{{Log-probability with the analogy:}} {lp_s1:.1f} nats\\\\\n"
            f"\\textbf{{Increase:}} {lp_s1 - lp_s0:+.1f} nats\n\\end{{ctxbox}}")
    row2 = [box("Baseline input: the target domain and the name of the invented concept",
                R.ctx_invention_target_only(ta, tp, t3["invention"])[1]["content"], fact, lp_b0),
            ctrl,
            box("Analogy input: both paths, their alignment, and the source concept's facts",
                R.ctx_invention_full(sa, ta, sp, tp, t3["projected"], src, t3["invention"])[1]["content"], fact, lp_b1)]

    model = rec["model"].split("_", 1)[1]
    cap = config["caption"].format(model=model, u=u, v=v, gain1=lp_src - lp_rel, gain2=lp_b1 - lp_b0)
    def three(boxes):
        return ["\\begin{minipage}[t]{0.32\\textwidth}\\vspace{0pt}", boxes[0], "\\end{minipage}\\hfill",
                "\\begin{minipage}[t]{0.32\\textwidth}\\vspace{0pt}", boxes[1], "\\end{minipage}\\hfill",
                "\\begin{minipage}[t]{0.32\\textwidth}\\vspace{0pt}", boxes[2], "\\end{minipage}"]
    tex = ["\\begin{figure*}[t]", "\\centering", "\\scriptsize",
           "\\textbf{(a) Predicting the target path of the analogy}\\par\\smallskip", *three(row1),
           f"\\par\\smallskip\\textbf{{Increase over the baseline: control {lp_fmt - lp_rel:+.1f} nats, real source path {lp_src - lp_rel:+.1f} nats}}\\par\\bigskip",
           "\\textbf{(b) Predicting a fact about the invented concept}\\par\\smallskip", *three(row2),
           f"\\par\\smallskip\\textbf{{Increase over the baseline: invented fact {lp_b1 - lp_b0:+.1f} nats, existing source fact {lp_s1 - lp_s0:+.1f} nats}}",
           f"\\caption{{\\ai{{{cap}}}}}", "\\label{fig:simple}", "\\end{figure*}"]
    text = "\n".join(tex) + "\n"
    (out / "fig_simple.tex").write_text(text)
    Path(config["paper_media_dir"], "fig_simple.tex").write_text(text)
    summary = {"record": rid, "logp_relations_only": lp_rel, "logp_format_control": lp_fmt, "logp_with_source": lp_src, "fact": fact,
               "source_fact": sfact, "logp_source_fact_without": lp_s0, "logp_source_fact_with": lp_s1,
               "logp_target_only": lp_b0, "logp_with_analogy": lp_b1}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true", help="never used by Claude; standing rule")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
