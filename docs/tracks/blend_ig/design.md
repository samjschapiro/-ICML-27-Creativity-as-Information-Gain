# blend_ig: information gain licensed by a conceptual blend

The blending counterpart of the analogy track (`docs/tracks/analogy_ig/design.md`). The analogy
track ended with one measurement design the user accepted: a fixed task with a relational
skeleton, the whole description of the new concept scored jointly, and a ladder of inputs that
reveal more of the creative structure step by step. This track applies the same design to
KOMBINE blends. Written before scoring (2026-09-19); predictions are fixed here.

## Theory in one paragraph

A blend of two concepts u and v is a third concept c built on a shared schema g (the generic
space) that both u and v instantiate. Its description is a set of triples, each tagged by where
its structure comes from: from u only, from v only, from both at once (a slot both inputs feed,
tag "uv"), or from neither (true of the blend alone, tag "emergent"). In code-length terms
(`docs/research_context.md`): g is a program shared by the two inputs, the blend is a third run
of that program with a slot fed by both parameter sets, and the emergent triples are the part of
the blend that neither input's description predicts but g together with both inputs does. The
paper (`sections/05_methods.tex`) names two gains: joint compression gain (the shared schema makes
the blend's inherited properties more likely than either input alone did) and emergent property
gain (the same for the emergent properties).

## Data

`comb-creat-eval/data/kg_creat/kombine_test30`, mode `blending`: 35 models x 30 cross-domain
pairs, one blend each, 1033 blends. Per blend: concept name c, generic space g (one phrase),
4 to 6 triples with head c and a tag in {u, v, uv, emergent}, and the generator's list of
emergent inferences. Verdicts: a 3-judge panel (haiku-4.5, gpt-5.4, o3) on generic_ok (is g a
real, specific schema both inputs instantiate), coherent (do the triples describe one coherent
concept), scope in {1, 2, 3} (single-scope, double-scope, double-scope with emergent structure).
Upstream aggregates: generic_ok and blend_utility are majorities, blend_integration is the median
scope. Judges corrected some tags in their explanations but the record keeps the generator's tags;
we use the generator's tags and say so.

Counts (2026-09-19): valid blend V_bl := majority generic_ok and coherent and scope >= 2, 391 of
1033; unanimous on all three, 156; unanimous scope 3, 187; upstream generic_ok true, 453.

Pairs are the same 30 anchor pairs as the analogy items (prompt ids F0..F29 vs E0..E29), which
allows a blend-vs-analogy comparison per pair later.

## Measurement

Scored block: the whole description of the blend, every triple rendered as one line
`c --relation--> tail` in the generator's order. Task text (fixed for every input):

    Task: Your task is to describe a concept with relational triples, one per line, in the form
    'head --relation--> tail'. Describe the novel concept '<c>' using these relations in this
    order, where X, Y, Z stand for concepts you must supply:
    <c> --grants--> X
    <c> --requires--> Y
    ...

Seven inputs appended to the task; the first six form a ladder in order of how much of the blend they reveal, the seventh is the schema without the inputs:

| input | appended text |
|---|---|
| skeleton | nothing |
| input u | "<c> is a new concept built from <u>." |
| input v | "<c> is a new concept built from <v>." |
| inputs | "<c> is a new concept built by blending <u> and <v>." |
| vacuous schema (fixed control) | inputs + " What <u> and <v> share, and what makes them blendable, is: both exist and involve change." |
| generic space | inputs + " What <u> and <v> share, and what makes them blendable, is: <g>." |
| generic space alone | "<c> is a new concept built on this shared schema: <g>." |

The vacuous schema is the fixed control for the generic space: the judge instructions name
"both exist" and "both involve change" as the canonical vacuous schemas, so this is the
strongest content-free stand-in for g in the same slot. No per-item control. No system message.
Per-token log-probabilities are attributed to entity or relation spans of each triple, so every
quantity below exists for the whole block and for each triple.

Quantities, L = log-probability of the block (or of one triple) under an input:
- second-input gain: L(inputs) - max(L(input u), L(input v)). Does the blend need both inputs.
- generic-space gain: L(generic) - L(inputs). What the shared schema adds once both inputs are
  named. Control: L(vacuous) - L(inputs).
- total gain: L(generic) - L(skeleton).
- joint compression gain (paper, user's framing 2026-09-19: "the generic space makes the blended
  space's properties from each input u or v more likely than either of the input concepts did on
  their own"): on the triples tagged u or v, L(generic) - max(L(input u), L(input v)), sums taken
  within the blend before the max. The uv triples are reported separately (shared-slot gain) and
  folded in as a variant.
- emergent property gain (paper: "the generic space makes the blended concept's property more
  likely than either of the input concepts did on their own"): the same contrast on the emergent
  triples.
- both paper gains are reported under two readings of "the generic space": the schema with the
  two inputs named (generic) and the schema phrase alone (generic space alone).
- synergy per triple: S(t) = [L(t | inputs) - L(t | skeleton)] - [L(t | u) - L(t | skeleton)] -
  [L(t | v) - L(t | skeleton)]. Positive when the two inputs together predict the triple more
  than the sum of what each does alone (the interaction-information proxy for the PID story in
  the theory notes).

## Predictions (fixed before scoring)

- P1 (both inputs needed): second-input gain > 0 for most valid blends, every reader.
- P2 (the schema carries information): generic-space gain > 0 for most valid blends and larger
  than the vacuous control, every reader. Falsified if the real schema is worth no more than
  "both exist and involve change".
- P3 (tag structure): for u-tagged triples the gain from input u alone is close to the gain from
  both inputs, and input v adds little; symmetric for v-tagged triples. For uv-tagged and
  emergent triples both inputs are needed: gain from both inputs well above the better single
  input. Synergy S ordered emergent >= uv > u, v on average. Falsified if the tags do not
  organise the gains at all.
- P4 (judges): generic-space gain is larger when the panel unanimously accepts g than when it
  unanimously rejects it; the emergent-triple generic-space gain is larger for unanimous scope 3
  than unanimous scope 1. Coherence, as before, is expected to be invisible at the triple level
  beyond the absolute log-probability.
- P5 (surprise): report the correlation of surprise (KOMBINE R) with the gains; no direction
  fixed, since for analogies surprise lowered alignment gain and slightly raised invention gain.
- P6 (blend vs analogy): report the total gain per triple next to the invention gain per triple
  on the same 30 pairs. The theory says blending is program induction and analogy is alignment,
  so the schema is expected to buy less per triple than the analogy mapping did.

## Readers

Same six as the analogy track: Qwen2.5-7B (mlx, local), Llama-3.1-8B, Mistral-7B-v0.3,
Gemma-2-9B, Qwen2.5-14B, OLMo-2-7B (transformers on a Lambda A100). All 1033 blends are scored
so every judge contrast has both polarities; headline numbers are over the 391 valid blends.

## Caveats to carry

- Tags are the generator's; judges disagreed with some. A tag-level result is therefore noisy in
  the direction of weakening P3.
- The inputs are names only (as the generator saw them); the reader's knowledge of u and v is
  the world model. Nothing describes u or v in the prompt.
- Some blends name things that exist (a "liquid franchise" is close to liquid democracy), so a
  reader may know the blend already; originality is not controlled here.
- The generic space is free text of varying quality; a low gain may mean a poor phrase rather
  than a poor schema. The judge's generic_ok verdict is the check.
