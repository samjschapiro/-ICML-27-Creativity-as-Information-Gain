# analogy_ig: empirical tests of the analogy information-gain taxonomy

Theory: `docs/memos/2026-09-18_analogy_information_gain_types.md`. Five types of gain an analogy
licenses: (1) alignment, (2) candidate inference, (3) concept invention, (4) schema abstraction,
(5) restructuring. Each is a distinct conditional code length. This track turns each into a
falsifiable prediction on the KOMBINE analogy data.

## Data

`comb-creat-eval/data/kg_creat/kombine_test30`: 35 models x 30 cross-domain items, one analogy
each = 1037 heads with an artifact. Per head: p_u, p_v (aligned paths sharing a relation
sequence), projected source concept phi, its source triples Phi, the invention h, and the
projected image triples M[Phi]. Verdicts: U_an (structural gates + single-judge factuality) and a
3-judge panel (haiku-4.5, gpt-5.4, o3) on integration quality J^qua and utility J^utl.

Counts (2026-09-18): U_an = 1 for 561; of those, panel unanimous True on both dimensions for 232,
covering all 30 items. Invalid heads by reason: factual 281, relations_differ 135,
structures_overlap 32, discontinuous 25, revisits_node 3. Both polarities of each verdict are
available, which is what the contrasts below need.

## Probability model (reference agent)

All K(.) are estimated as -log p under a held-out local LM (teacher-forced, chat-formatted,
target as the assistant turn). Default `mlx-community/Qwen2.5-7B-Instruct-8bit` via mlx-lm.
Rule: never one of the 35 scored systems. Report a second model (Llama-3.1-8B) for sensitivity.
Per-token log-probs are attributed to entity vs relation spans of the rendered target.

## Phase 1 (local, free): types 1 and 3

### Type 1, alignment gain. Target: p_v.

| Context | Conditions on | Estimates |
|---|---|---|
| C0 | v, hop count | K(p_v \| v) |
| C_rho | v, relation skeleton rho | K(p_v \| v, rho) |
| C1 | v, u, true p_u (asks for the aligned path) | K(p_v \| p_u, alignment) |
| C1_shuf | v, u, donor path = true rho with another item's source entities | control |

Quantities (entity-token sums unless stated):
- G_skel = logp(C_rho) - logp(C0) on relation tokens: trivial for valid analogies; sanity check.
- G_M = logp(C1) - logp(C_rho) on entity tokens: does the source dictionary predict the target
  entities beyond the skeleton. This is the alignment gain proper.
- G_ctrl = logp(C1_shuf) - logp(C_rho): any-entities control.

Predictions:
- P1a: G_M > 0 and G_M > G_ctrl for U_an = 1. Falsified if the true source entities predict p_v no
  better than a donor's entities.
- P1b: G_M is larger for U_an = 1 than for relations_differ heads (mapping is not an isomorphism).
- P1c: G_M grows with n_hops (more skeleton, more shared structure). Weak prediction; report.

### Type 3, concept-invention gain. Target: each image triple (h, r_j, M(s_j)).

| Context | Conditions on | Estimates |
|---|---|---|
| B0 | v, p_v, bare name h | K(t \| D_v, name) |
| B1 | u, v, p_u, p_v, alignment, phi, Phi, h | K(t \| Phi, M) |
| S0 | u, p_u, phi (target = source triple) | K(t_src \| D_u) |
| S1 | full B1 context (target = source triple) | K(t_src \| everything) |

Quantities:
- E_inv(t) = logp(t \| B1) - logp(t \| B0), averaged over a record's image triples.
- A(t) = logp(t \| B1) / n_tokens: absolute plausibility under the full context.
- E_src(t) = logp(S1) - logp(S0) for source triples: known facts, control.

Predictions:
- P3a: E_inv separates integration = True from False among unanimous verdicts. J^qua asserts
  K(h \| Phi, M) ~ 0, which is exactly high E_inv.
- P3b: A separates utility = True from False among unanimous verdicts. Coherence is probability
  under a world model, not gain.
- P3c: dissociation. In a 2x2 (integration x utility, unanimous cells), E_inv tracks the
  integration axis and A tracks the utility axis; each is a weak predictor of the other axis.
  Falsified if one scalar explains both verdicts.
- P3d: E_inv >> E_src. A definition gains bits from the mapping; a known fact does not. This is the
  fact-vs-definition distinction between types 2 and 3.

### Dissociability across types
- Correlate G_M (type 1) with mean E_inv (type 3) over the 232 strict records. The taxonomy claims
  different objects; a high correlation beyond what shared validity induces would mean one
  underlying factor, which is the thing the paper says is not the case.

## Phase 2 (API, budgeted): types 4, 2, 5

### Type 4, schema abstraction (transfer to a third domain)
For each strict record, take rho and 3 held-out entities w from domains different from u and v.
Ask a generator (not the probability model) to instantiate a path from w with exactly rho; gate
with the factuality judge. Transfer rate T(rho) = fraction of w with a factual instantiation.
Controls: rho from relations_differ heads (not a real isomorphism) and length-matched random
relation sequences drawn from the pool. Prediction P4: T(valid rho) > T(random rho); and the
LM-probability version logp(path_w \| w, rho) - logp(path_w \| w) > 0 on the generated paths.
Cost: ~232 x 3 generations + judge calls; small.

### Type 2, candidate inference (constructed, since the prompt excludes it)
For each strict record, collect source triples (a_i, r, a_j) with both endpoints in dom(M) from a
generator asked to list facts about the source entities; project M[t]; factuality-judge M[t] on
the target. Prediction P2: a nontrivial fraction is factual, and E_infer = logp(M[t] \| B1) -
logp(M[t] \| B0) is high for both true and false projections (gain is not truth). Contrast with
type 3: type-2 images have higher absolute logp(B0) than type-3 images (existing target entities).

### Type 5, restructuring (prior shift in the target)
Sample k = 10 short relational descriptions of v with and without the analogy in context; measure
the shift as mean logp(before-samples \| after-context) - logp(before-samples \| before-context)
and as embedding-space displacement toward p_u's relations. Control: mismatched analogy.
Lowest priority; noisiest.

## Caveats to carry into the paper
- LM log-probs are model-dependent upper bounds on K; report two models.
- 8-bit quantisation perturbs log-probs; contrasts are within-model so this cancels to first order.
- Prompt wording is a nuisance variable; contexts differ only in the conditioning content.
- Judge verdicts are themselves LLM outputs; unanimity is the filter, human corroboration in the
  KOMBINE appendix (75% on unanimous items) is the ceiling.
