"""Every prompt the blend track shows a reader.

Scored block: the whole description of the blend, one triple per line, rendered exactly like the
analogy track's invention block. Task: fixed relational skeleton with the blend's name kept and
every other entity replaced by a letter. Inputs: appended sentences that reveal the blend's
inputs and generic space step by step. No system message.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analogy_ig.render import invention_skeleton, render_path, render_triple

VACUOUS_SCHEMA = "both exist and involve change"


def blend_task(concept: str, triples: list) -> str:
    skel_rows = invention_skeleton(concept, triples)
    skel, _ = render_path(skel_rows)
    letters = ", ".join(sorted({x for row in skel_rows for x in (row[0], row[2]) if x != concept}))
    return ("Task: Your task is to describe a concept with relational triples, one per line, in the "
            f"form 'head --relation--> tail'. Describe the novel concept '{concept}' using these "
            f"relations in this order, where {letters} stand for concepts you must supply:\n{skel}")


def render_block(triples: list) -> tuple[str, list[tuple[int, int, str]]]:
    """Like render_path, but every span is labelled '<line index>:<kind>' so the scorer's per-kind
    sums give per-triple log-probabilities as well."""
    out, spans = "", []
    for i, triple in enumerate(triples):
        line, sp = render_triple(triple)
        base = len(out)
        spans.extend((a + base, b + base, f"{i}:{k}") for a, b, k in sp)
        out += line + ("\n" if i < len(triples) - 1 else "")
    return out, spans


def blk_input(concept: str, x: str) -> str:
    return f"{concept} is a new concept built from {x}."


def blk_inputs(concept: str, u: str, v: str) -> str:
    return f"{concept} is a new concept built by blending {u} and {v}."


def blk_schema(concept: str, u: str, v: str, schema: str) -> str:
    return blk_inputs(concept, u, v) + f" What {u} and {v} share, and what makes them blendable, is: {schema}."


def ctx_blend_block(task: str, block: str | None) -> list[dict]:
    return [{"role": "user", "content": task if block is None else task + "\n\n" + block}]


def all_inputs(rec: dict) -> dict[str, list[dict]]:
    """The six inputs of the ladder, in order, for one blend record."""
    c, u, v, g = rec["concept"], rec["u"], rec["v"], rec["generic_space"]
    task = blend_task(c, rec["triples"])
    return {
        "skeleton": ctx_blend_block(task, None),
        "input_u": ctx_blend_block(task, blk_input(c, u)),
        "input_v": ctx_blend_block(task, blk_input(c, v)),
        "inputs": ctx_blend_block(task, blk_inputs(c, u, v)),
        "vacuous": ctx_blend_block(task, blk_schema(c, u, v, VACUOUS_SCHEMA)),
        "generic": ctx_blend_block(task, blk_schema(c, u, v, g)),
    }
