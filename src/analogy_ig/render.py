"""Text rendering of KOMBINE artifacts and the conditioning contexts for each information-gain type.

Every context is a list of chat messages; every target is a string plus labelled character spans
so per-token log-probs can be attributed to entities vs relations.
"""

from __future__ import annotations

SEP_REL = " --"
SEP_END = "--> "


def render_triple(triple: list) -> tuple[str, list[tuple[int, int, str]]]:
    """'head --relation--> tail' with (start, end, kind) spans, kind in {entity, relation}."""
    h, r, t = (str(x) for x in triple)
    s = ""
    spans = []
    spans.append((len(s), len(s) + len(h), "entity")); s += h
    s += SEP_REL
    spans.append((len(s), len(s) + len(r), "relation")); s += r
    s += SEP_END
    spans.append((len(s), len(s) + len(t), "entity")); s += t
    return s, spans


def render_path(path: list) -> tuple[str, list[tuple[int, int, str]]]:
    """One triple per line; spans are offset into the whole string."""
    out, spans = "", []
    for i, triple in enumerate(path):
        line, sp = render_triple(triple)
        base = len(out)
        spans.extend((a + base, b + base, k) for a, b, k in sp)
        out += line + ("\n" if i < len(path) - 1 else "")
    return out, spans


def relation_sequence(path: list) -> list[str]:
    return [str(t[1]) for t in path]


def substitute_entities(path_relations: list[str], donor_path: list) -> list:
    """Build a path with the given relations but the donor's entities (same length required)."""
    if len(path_relations) != len(donor_path):
        raise ValueError("FATAL: donor path length mismatch")
    return [[donor_path[i][0], path_relations[i], donor_path[i][2]] for i in range(len(donor_path))]


# ---------------------------------------------------------------------------------------------
# Type 1: alignment gain. Target = p_v. Contexts vary what is known about the source.
# ---------------------------------------------------------------------------------------------

_SYS = ("You write factual relational paths. A path is a chain of triples, one per line, in the "
        "form 'head --relation--> tail', where each line's head is the previous line's tail.")


def ctx_align(v: str, u: str | None, n_hops: int, relations: list[str] | None,
              source_path: list | None) -> list[dict]:
    """C0: v only. C_rho: v + relation skeleton. C1: v + aligned source path (true or donor)."""
    if source_path is not None:
        src_txt, _ = render_path(source_path)
        user = (f"Here is a {n_hops}-triple path describing '{u}':\n{src_txt}\n\n"
                f"Write the analogous {n_hops}-triple path describing '{v}', using exactly the same "
                f"relations in the same order, starting at '{v}'.")
    elif relations is not None:
        rel_txt = ", ".join(f"'{r}'" for r in relations)
        user = (f"Write a {n_hops}-triple path describing '{v}', starting at '{v}', using exactly "
                f"these relations in this order: {rel_txt}.")
    else:
        user = f"Write a {n_hops}-triple path describing '{v}', starting at '{v}'."
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------------------------
# Type 3: concept-invention gain. Target = one image triple (h, r_j, M(s_j)).
# ---------------------------------------------------------------------------------------------

_SYS_FACT = "You state one relational fact as a single triple in the form 'head --relation--> tail'."


# In these four contexts "source" is the domain the projected concept comes from and "target"
# is the domain the invention lands in. KOMBINE lets the projection run in either direction
# (u -> v or v -> u); the caller decides which anchor/path is which (see score_logprob).


def ctx_invention_target_only(target: str, target_path: list, invention: str) -> list[dict]:
    """B0: target domain alone plus the bare name of the new concept."""
    pt_txt, _ = render_path(target_path)
    user = (f"Facts about '{target}':\n{pt_txt}\n\n"
            f"'{invention}' is a new concept in the domain of '{target}'. State one fact about "
            f"'{invention}' as a triple whose head is '{invention}'.")
    return [{"role": "system", "content": _SYS_FACT}, {"role": "user", "content": user}]


def ctx_invention_full(source: str, target: str, source_path: list, target_path: list,
                       projected: str, source_triples: list, invention: str) -> list[dict]:
    """B1: both paths, the alignment, the projected source concept and its source structure."""
    ps_txt, _ = render_path(source_path)
    pt_txt, _ = render_path(target_path)
    pairs = "\n".join(f"'{a}' corresponds to '{b}'" for a, b in _alignment_pairs(source_path, target_path))
    src = "\n".join(render_triple(t)[0] for t in source_triples)
    user = (f"An analogy between '{source}' and '{target}'.\n\nPath for '{source}':\n{ps_txt}\n\n"
            f"Path for '{target}':\n{pt_txt}\n\nAlignment (position by position):\n{pairs}\n\n"
            f"'{projected}' is a concept from the domain of '{source}' with no counterpart in the "
            f"domain of '{target}'. Facts about '{projected}':\n{src}\n\n"
            f"Carrying '{projected}' across the alignment defines a new concept '{invention}' in "
            f"the domain of '{target}'. State one fact about '{invention}' as a triple whose head is "
            f"'{invention}'.")
    return [{"role": "system", "content": _SYS_FACT}, {"role": "user", "content": user}]


def ctx_source_fact_only(source: str, source_path: list, projected: str) -> list[dict]:
    """Control for the definition-vs-fact contrast: a known source fact given the source domain."""
    ps_txt, _ = render_path(source_path)
    user = (f"Facts about '{source}':\n{ps_txt}\n\n'{projected}' is a concept in the domain of '{source}'. "
            f"State one fact about '{projected}' as a triple whose head is '{projected}'.")
    return [{"role": "system", "content": _SYS_FACT}, {"role": "user", "content": user}]


def ctx_source_fact_full(source: str, target: str, source_path: list, target_path: list, projected: str,
                         source_triples: list, invention: str, exclude: list) -> list[dict]:
    """Same full context as B1 but asking for a fact about the SOURCE concept.

    Leave-one-out: the scored source triple is removed from the listed facts, otherwise the
    context contains the target verbatim and the gain measures copying, not derivation.
    """
    remaining = [t for t in source_triples if t != exclude]
    if len(remaining) == len(source_triples):
        raise ValueError("FATAL: excluded triple not found among source triples")
    msgs = ctx_invention_full(source, target, source_path, target_path, projected, remaining, invention)
    msgs[-1]["content"] = msgs[-1]["content"].rsplit("State one fact about", 1)[0] + (
        f"State one fact about '{projected}' as a triple whose head is '{projected}'.")
    return msgs


def _alignment_pairs(path_u: list, path_v: list) -> list[tuple[str, str]]:
    pairs = []
    for tu, tv in zip(path_u, path_v):
        pairs.append((str(tu[0]), str(tv[0])))
    pairs.append((str(path_u[-1][2]), str(path_v[-1][2])))
    # dedupe while preserving order
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def projection_direction(path_u: list, path_v: list, source_triples: list, image_triples: list) -> str:
    """Which way the projection runs, from where its triples' tail entities live.

    Source triples describe the projected concept in its own domain, so their tails should sit on
    that domain's path; image tails should sit on the other path. Returns 'u_to_v', 'v_to_u', or
    'undetectable' (no tail on either path, or a tie).
    """
    def ents(path):
        out = set()
        for h, _, t in path:
            out.add(str(h).lower()); out.add(str(t).lower())
        return out
    eu, ev = ents(path_u), ents(path_v)
    u2v = sum(str(t[2]).lower() in eu for t in source_triples) + sum(str(t[2]).lower() in ev for t in image_triples)
    v2u = sum(str(t[2]).lower() in ev for t in source_triples) + sum(str(t[2]).lower() in eu for t in image_triples)
    if u2v > v2u:
        return "u_to_v"
    if v2u > u2v:
        return "v_to_u"
    return "undetectable"


def abstract_path(relations: list[str]) -> list:
    """A content-free path with the given relations: A --r1--> B, B --r2--> C, ..."""
    letters = "ABCDEFGHIJ"
    return [[letters[i], r, letters[i + 1]] for i, r in enumerate(relations)]


def ctx_align_format(v: str, n_hops: int, relations: list[str]) -> list[dict]:
    """Format control: the analogy prompt's exact shape with an abstract path in the source slot
    and no source concept named, so nothing can be reconstructed and nothing is false."""
    src_txt, _ = render_path(abstract_path(relations))
    user = (f"Here is a {n_hops}-triple path with these relations:\n{src_txt}\n\n"
            f"Write the analogous {n_hops}-triple path describing '{v}', using exactly the same "
            f"relations in the same order, starting at '{v}'.")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


def ctx_align_named(v: str, u: str, n_hops: int, relations: list[str]) -> list[dict]:
    """Named-source control: the reader is told which concept the analogy comes from, but is not
    shown its path. Otherwise identical to the relations-only baseline."""
    rel_txt = ", ".join(f"'{r}'" for r in relations)
    user = (f"Write a {n_hops}-triple path describing '{v}' by analogy with '{u}', starting at '{v}', "
            f"using exactly these relations in this order: {rel_txt}.")
    return [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------------------------
# Invention measurement framed like the alignment one: "describe the novel concept using these
# relations". The skeleton fixes relations, fact count and the concept's slot; entities are supplied.
# No system message; the task text carries the role instruction.
# ---------------------------------------------------------------------------------------------

_LETTERS = "XYZWVUTSRQ"


def invention_skeleton(invention: str, image_triples: list) -> list:
    """Image triples with every entity other than the invented concept replaced by a variable,
    consistently (the same entity gets the same letter wherever it recurs)."""
    var, out = {}, []
    for h, r, t in image_triples:
        row = []
        for e in (str(h), str(t)):
            if e == invention:
                row.append(e)
            else:
                if e not in var:
                    var[e] = _LETTERS[len(var)]
                row.append(var[e])
        out.append([row[0], r, row[1]])
    return out


def invention_task(invention: str, target_anchor: str, image_triples: list) -> str:
    skel, _ = render_path(invention_skeleton(invention, image_triples))
    letters = ", ".join(sorted({x for row in invention_skeleton(invention, image_triples) for x in (row[0], row[2]) if x != invention}))
    return ("Task: Your task is to describe a concept with factual relational triples, one per line, in "
            f"the form 'head --relation--> tail'. Describe the novel concept '{invention}', a concept in "
            f"the domain of '{target_anchor}', using these relations in this order, where {letters} stand "
            f"for concepts you must supply:\n{skel}")


def ctx_invention_block(task: str, block: str | None) -> list[dict]:
    return [{"role": "user", "content": task if block is None else task + "\n\n" + block}]


def inv_block_instruction(projected: str, source_anchor: str) -> str:
    return f"Use an analogy with the concept {projected} from the domain of {source_anchor}."


def inv_block_content(projected: str, source_anchor: str, target_anchor: str, source_triples: list,
                      source_path: list, target_path: list) -> str:
    facts, _ = render_path(source_triples)
    ps, _ = render_path(source_path); pt, _ = render_path(target_path)
    pairs = "; ".join(f"'{a}' corresponds to '{b}'" for a, b in _alignment_pairs(source_path, target_path))
    return (f"Use the analogous {projected} facts:\n{facts}\n\nand the analogy between {source_anchor} and "
            f"{target_anchor}:\n{ps}\n\n{pt}\n\nwhere {pairs}.")


def inv_block_mapping(projected: str, source_anchor: str, target_anchor: str,
                      source_path: list, target_path: list) -> str:
    """Intermediate input: the instruction plus the full analogy (both paths with correspondences),
    but not the facts about the source concept."""
    ps, _ = render_path(source_path); pt, _ = render_path(target_path)
    pairs = "; ".join(f"'{a}' corresponds to '{b}'" for a, b in _alignment_pairs(source_path, target_path))
    return (f"Use an analogy with the concept {projected} from the domain of {source_anchor}, following "
            f"the analogy between {source_anchor} and {target_anchor}:\n{ps}\n\n{pt}\n\nwhere {pairs}.")
