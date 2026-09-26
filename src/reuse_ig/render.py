"""Prompts for the reuse track. Formats follow the analogy track (system message _SYS for paths)
and the invention/blend tasks (no system message). See docs/tracks/reuse_ig/design.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analogy_ig.render import _SYS, _alignment_pairs, render_path, render_triple, invention_skeleton
from src.blend_ig.render import blend_task, blk_inputs, blk_schema, render_block


def _pairs_txt(path_u, path_v, n_entities=None) -> str:
    pairs = _alignment_pairs(path_u, path_v)
    if n_entities is not None:
        pairs = pairs[:n_entities]
    return "; ".join(f"'{a}' corresponds to '{b}'" for a, b in pairs)


def analogy_block(u, v, path_u, path_v) -> str:
    pu, _ = render_path(path_u); pv, _ = render_path(path_v)
    return f"Here is an analogy between {u} and {v}:\n{pu}\n\n{pv}\n\nwhere {_pairs_txt(path_u, path_v)}."


# ---- 1a. held-out target hop ----------------------------------------------------------------

def heldout_hop(u, v, path_u, path_v) -> dict:
    n = len(path_v); known, last = path_v[:-1], path_v[-1]
    known_txt, _ = render_path(known); r_n = str(last[1])
    base = (f"Here are the first {n - 1} triples of a {n}-triple path describing '{v}':\n{known_txt}\n\n"
            f"Write the final triple, whose relation is '{r_n}'.")
    src_txt, _ = render_path(path_u)
    with_src = (f"Here is a {n}-triple path describing '{u}':\n{src_txt}\n\n"
                f"Here are the first {n - 1} triples of the analogous path describing '{v}', where "
                f"{_pairs_txt(path_u, path_v, n_entities=n)}:\n{known_txt}\n\n"
                f"Write the final triple of the path describing '{v}', whose relation is '{r_n}'.")
    target, spans = render_triple(last)
    ctx = lambda user: [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
    return {"target": target, "spans": spans, "contexts": {
        "target_so_far": ctx(base),
        "instruction": ctx(base + f" Do so by analogy with '{u}'."),
        "source_path": ctx(with_src)}}


# ---- 1b. held-out projected fact ------------------------------------------------------------

def heldout_fact(invention, target_anchor, projected, source_anchor, source_path, target_path,
                 source_triples, image_triples) -> dict:
    known, last = image_triples[:-1], image_triples[-1]
    known_txt, _ = render_path(known)
    task = (f"Here are {len(known)} facts about the novel concept '{invention}', a concept in the domain of "
            f"'{target_anchor}':\n{known_txt}\n\nState one more fact of the form '{invention} --{last[1]}--> X', "
            f"where X is a concept you must supply.")
    ps, _ = render_path(source_path); pt, _ = render_path(target_path); facts, _ = render_path(source_triples)
    mapping = (f"Use an analogy with the concept {projected} from the domain of {source_anchor}, following the "
               f"analogy between {source_anchor} and {target_anchor}:\n{ps}\n\n{pt}\n\nwhere {_pairs_txt(source_path, target_path)}.")
    content = (f"Use the analogous {projected} facts:\n{facts}\n\nand the analogy between {source_anchor} and "
               f"{target_anchor}:\n{ps}\n\n{pt}\n\nwhere {_pairs_txt(source_path, target_path)}.")
    target, spans = render_triple(last)
    ctx = lambda block: [{"role": "user", "content": task if block is None else task + "\n\n" + block}]
    return {"target": target, "spans": spans, "contexts": {
        "known_facts": ctx(None),
        "instruction": ctx(f"Use an analogy with the concept {projected} from the domain of {source_anchor}."),
        "mapping": ctx(mapping),
        "content": ctx(content)}}


# ---- 2a. within-task reuse: demonstrations before the alignment inputs ----------------------

def demo_block(demos: list[dict]) -> str:
    parts = []
    for d in demos:
        pu, _ = render_path(d["path_u"]); pv, _ = render_path(d["path_v"])
        parts.append(f"{pu}\n\n{pv}\n\nwhere {_pairs_txt(d['path_u'], d['path_v'])}.")
    head = ("Here is an example of an analogy: two paths using the same relations in the same order:" if len(demos) == 1
            else f"Here are {len(demos)} examples of analogies, each two paths using the same relations in the same order:")
    return head + "\n\n" + "\n\n".join(parts)


def within_task(v, u, path_u, path_v, demos1: list[dict], demos3: list[dict]) -> dict:
    n = len(path_v); rels = [str(t[1]) for t in path_v]
    rel_txt = ", ".join(f"'{r}'" for r in rels)
    rho = (f"Write a {n}-triple path describing '{v}', starting at '{v}', using exactly these relations "
           f"in this order: {rel_txt}.")
    src_txt, _ = render_path(path_u)
    c1 = (f"Here is a {n}-triple path describing '{u}':\n{src_txt}\n\nWrite the analogous {n}-triple path "
          f"describing '{v}', using exactly the same relations in the same order, starting at '{v}'.")
    target, spans = render_path(path_v)
    ctx = lambda user: [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
    return {"target": target, "spans": spans, "contexts": {
        "relations_demo1": ctx(demo_block(demos1) + "\n\n" + rho),
        "source_demo1": ctx(demo_block(demos1) + "\n\n" + c1),
        "relations_demo3": ctx(demo_block(demos3) + "\n\n" + rho),
        "source_demo3": ctx(demo_block(demos3) + "\n\n" + c1)}}


# ---- 2b. cross-task, same pair ---------------------------------------------------------------

def analogy_to_blend(blend: dict, analogy: dict, other: dict) -> dict:
    """Blend description under: told it blends u and v, plus the pair's analogy / plus analogy and
    generic space / plus an analogy between two other concepts."""
    c, u, v, g = blend["concept"], blend["u"], blend["v"], blend["generic_space"]
    task = blend_task(c, blend["triples"])
    same = analogy_block(analogy["u"], analogy["v"], analogy["path_u"], analogy["path_v"])
    oth = analogy_block(other["u"], other["v"], other["path_u"], other["path_v"])
    target, spans = render_block(blend["triples"])
    ctx = lambda block: [{"role": "user", "content": task + "\n\n" + block}]
    return {"target": target, "spans": spans, "contexts": {
        "inputs_analogy": ctx(blk_inputs(c, u, v) + " " + same),
        "inputs_analogy_generic": ctx(blk_schema(c, u, v, g) + " " + same),
        "inputs_other_analogy": ctx(blk_inputs(c, u, v) + " " + oth)}}


def blend_to_analogy(v, u, path_u, path_v, generic_space: str, other_generic: str, other_pair: tuple) -> dict:
    n = len(path_v); rels = [str(t[1]) for t in path_v]
    rel_txt = ", ".join(f"'{r}'" for r in rels)
    rho = (f"Write a {n}-triple path describing '{v}', starting at '{v}', using exactly these relations "
           f"in this order: {rel_txt}.")
    src_txt, _ = render_path(path_u)
    c1 = (f"Here is a {n}-triple path describing '{u}':\n{src_txt}\n\nWrite the analogous {n}-triple path "
          f"describing '{v}', using exactly the same relations in the same order, starting at '{v}'.")
    same = f"'{u}' and '{v}' share this schema: {generic_space}."
    oth = f"'{other_pair[0]}' and '{other_pair[1]}' share this schema: {other_generic}."
    target, spans = render_path(path_v)
    ctx = lambda user: [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
    return {"target": target, "spans": spans, "contexts": {
        "relations_generic": ctx(same + "\n\n" + rho),
        "source_generic": ctx(same + "\n\n" + c1),
        "relations_other_generic": ctx(oth + "\n\n" + rho)}}
