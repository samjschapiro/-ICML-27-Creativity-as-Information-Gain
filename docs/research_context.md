# Research Context

**Paper:** [ICML 27] Creativity as Information Gain: A Theoretical Perspective
**Overleaf clone:** `papers/6aad5b22035168dccf225738/` (skeleton only as of 2026-09-18)
**Author:** Samuel Schapiro (first author on the KOMBINE paper below)

## Goal

Build a theoretical account of combinatorial creativity (association, analogy,
blending) in algorithmic information theory terms, by lining up two papers:

1. **KOMBINE** (Schapiro et al., ICLR 27 preprint): an open-ended benchmark
   scoring associations, analogies, and blends between distant entities on
   utility, surprise, and originality, plus "emergent creativity" of the
   invented concept. 35 LLMs, 200 humans. Key empirical findings: LLMs are
   ~10pp better at analogy than blending; the bottleneck to blending is finding
   a valid generic space (abstraction), since conditioned on a valid generic
   space models describe a coherent blend 97% of the time; association
   originality is the best cross-task predictor; same-family models produce
   "inventive multiples".
2. **Elmoznino et al. 2024/2025** (arXiv 2410.14817), "A Complexity-Based
   Theory of Compositionality": representational compositionality
   C(Z) = K(Z) / K(Z|W), where K(Z) = min K(p_w) + K(W|p_w) + K(f) + K(Z|W,f).
   Sentences W are discrete parts, f is the semantics mapping parts to
   representation, and a compositional representation is expressive but
   describable as a *simple* f of parts. Estimated via prequential coding and
   log-prob under learned models as upper bounds on conditional K.

## The bridge (user's framing, 2026-09-18)

Shared hidden variable: a program mapping discrete parts to structure.
Elmoznino's semantics f corresponds to KOMBINE's structure mapping M (analogy)
and generic space g (blending).

### 1. Structure mapping as compression
- A valid analogy of length n = shared relational skeleton rho = (r_1..r_n)
  plus substitution dictionary M: a_i -> b_i. Joint description length:
  K(p_u, p_v) <= K(rho) + K(entities of p_u) + K(M) + K(p_v | M[p_u]).
  U_an = 1 iff the last (reconstruction) term is zero. Savings ~ algorithmic
  mutual information I(p_u : p_v) ~ K(rho). Gentner's systematicity principle
  falls out as an MDL preference (structure transferred per dictionary bit).
- The three KOMBINE facets are code lengths under three reference models:
  - Utility = reconstruction error under a world model (0 iff all triples
    factual; for blends, iff g reconstructs both inputs).
  - Surprise = code length of the dictionary under a semantic-proximity prior,
    sum_i -log p_sem(b_i | a_i), monotone in averaged cosine distance.
  - Originality = code length under the empirical response distribution,
    -log p_pop(artifact), estimated by nearest-neighbor distance to the pool.
- Signature of a creative mapping: low K(M), high code length under priors.
  Suggests analogical creativity as a ratio: bits of target structure licensed
  (matched path + projected invention) over bits to specify the mapping.

### 2. Emergent creativity as information gain
- Analogy: h = M[Phi]. EC_an(h) = K(h | D_v) - K(h | D_v, D_u, M)
  = I(h : D_u | D_v). Existing judges decompose it: J^qua_an asserts
  K(h | Phi, M) ~ 0; O_an(h) is the population-code-length version of the first
  term; utility (coherence) gates that h has nonzero probability under a world
  model of D_v. Bayesian counterpart: Bayesian surprise (Itti & Baldi; Varshney
  2019). Dynamic version: Schmidhuber compression progress.
- Blending: g is a shared program, D_u ~ g(theta_u), D_v ~ g(theta_v)
  (Elmoznino's modular f, Example 5, reused across domains).
  K(D_u, D_v) <= K(g) + K(theta_u) + K(theta_v) + errors; J^gen_bl = 1 iff both
  error terms small. Blend = third instantiation g(theta_uv) with at least one
  slot fed by both parameter sets (the "uv" triple). Emergent set
  Delta = g(theta_uv) \ (D_u u D_v): K(Delta | D_u, D_v) high but
  K(Delta | g, theta_uv) ~ 0. This is Chalmers' weak emergence in algorithmic
  terms; the right notion is Bennett's logical depth, not K.
- Scope as partial information decomposition (PID) of I(c' ; D_u, D_v):
  u/v triples = unique info, uv = redundancy on relation + synergy on filler,
  Delta = pure synergy. Single-scope (1) = zero synergy (concatenation, maximally
  disentangled f); double-scope (2) = nonzero synergy; emergent (3) = synergy
  surviving removal of everything projectable. J^qua in {1,2,3} becomes graded:
  emergence = synergistic fraction.

### 3. Claims to put in the paper
- Why blending is harder than analogy: analogy is alignment (find an
  isomorphism between two existing paths); blending is program induction
  (invent g that compresses both domains), the same intractable joint
  minimization Elmoznino approximate with a discrete autoencoder.
- Tension between frameworks: Elmoznino's compositionality rewards
  disentanglement (n-gram size 1) since it minimizes K(f); emergent creativity
  requires context-sensitivity (the uv slot is where parts interact; Delta is
  what independence cannot produce). So combinatorial creativity is not
  monotone in representational compositionality. Association lives near pure
  systematicity, blending in the interaction terms (sharpens KOMBINE Fig. 3).

### 4. Estimators
- Per-triple LLM log-prob emergence score:
  E(t) = log p(t | u, v, g) - max{ log p(t | u), log p(t | v) }.
  Emergent triples have large E, inherited ~0, incoherent low absolute
  log p(t | u, v, g) (recovers utility). Average over a blend gives a continuous
  emergence score to validate against J^qua and the ICC = 0.67 panel. For
  analogy, condition on (D_v, D_u, M) vs D_v alone.
- Prequential coding on the generic space: how fast a small model learns
  D_u u D_v given g vs without, estimating K(g) and the compression it buys.
  Direct test of "abstraction as shared program".
- Caveats to state: PID has no canonical synergy measure (pick BROJA or I_min,
  report sensitivity); LLM log-probs are model-dependent upper bounds, so use a
  held-out probability model, not any of the 35 scored systems; the gain is over
  the conceptual space (hypotheses), not the world, so coherence must gate it.

## Current state (2026-09-19)

- Track `analogy_ig` built and run: alignment gain and invention gain estimated for 561 valid
  KOMBINE analogies under six held-out readers (Qwen2.5-7B/14B, Llama-3.1-8B, Mistral-7B,
  Gemma-2-9B, OLMo-2-7B). Headline: alignment gain +13.1 nats (positive for 92.5%), invention gain
  +18.2 nats vs -5.2 for existing facts (99.6%); the two gains uncorrelated; integration judge
  visible to 5/6 readers, coherence judge to none; naming the source concept without its path adds
  nothing. Report: `docs/reports/2026-09-18_analogy_ig_phase1/`.
- Interpretation settled with the user: alignment gain = the target path's description length
  saved by describing it in terms of the source path, i.e. algorithmic mutual information of the
  two paths given the shared relation sequence; the message is the analogy-maker's specific path,
  not the domain label. Surprise (dictionary cost) trades against it.
- Paper (Overleaf, standard sections): user's Figma overview is Figure 1; all figures are to be
  Figma; histogram PDFs provided for import. Text sections still to be written.
- Open: association-vs-analogy information test (needs association paths on the analogy pairs),
  symmetry of the gain, how to state the coherence-judge null, phase 2 (types 2, 4, 5).
