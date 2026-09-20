# blend_ig progress

## 2026-09-19: track built, predictions fixed, six readers scored

- Built the track as the blending counterpart of the accepted invention design (`design.md`):
  fixed skeleton task, whole description of the blend scored jointly, seven appended inputs
  (skeleton, input u, input v, both inputs, vacuous schema as fixed control, generic space, generic
  space alone). The user's framing of the two paper gains (2026-09-19): "the generic space makes
  the blended space's properties from each input u or v more likely than either of the input
  concepts did on their own" (joint compression gain, tags u and v) and the same for the emergent
  properties (emergent property gain). The both-inputs control was promoted to a figure panel at
  the user's request so the generic space's own contribution is isolated.
- Data: 1033 blends, 35 models, 30 pairs; valid V_bl (majority generic_ok, coherent, scope >= 2)
  393, all 30 pairs. Generator's tags used as given.
- Scored all 1033 under Qwen2.5-7B (local mlx, 12 min) and Llama-3.1-8B, Mistral-7B, Gemma-2-9B,
  Qwen2.5-14B, OLMo-2-7B (one A100, 35 min, instance auto-terminated by
  `lambda_autofinish_blend.sh`). Analysis: `analyze_blend_block.py` -> `downstream/blend_block_ladder/`.

### Results (393 valid blends, log-probability of the whole description averaged over readers)
- Ladder: skeleton -110.9, better single input -104.8, both inputs -102.5, vacuous schema -102.1,
  generic space -98.0. Second input +2.3 [1.9, 2.8] (69% positive, Wilcoxon p = 3e-21); generic
  space over both inputs +4.4 [3.9, 5.1] (80%, p = 1e-40); vacuous schema +0.3; generic space over
  vacuous +4.1 (77%, p = 3e-39); total +12.9 [12.1, 13.8] (95%, p = 3e-64).
- Per reader, generic-space gain over both inputs: Qwen-7B 4.8, Llama 3.5, Mistral 7.1, Gemma 1.2
  (p = 0.006), Qwen-14B 7.4, OLMo 2.7; all positive, all p < 0.01.
- Where the schema's information lands (generic space minus both inputs, per line, reader
  average): u lines -0.1, v lines +0.2, uv lines +2.4, emergent lines +0.2. The uv step is the
  largest for every one of the six readers (1.7 to 3.5). Inherited lines are predicted by the
  input that owns them: u lines +2.7 from Democracy-side input alone vs -1.8 from the other input,
  v lines +3.4 vs -1.0.
- Paper gains, generic space vs the better single input, summed within the blend:
  joint compression gain (u and v lines) +2.1 [1.7, 2.5], 70% positive; including uv lines +6.0
  [5.4, 6.7], 88%; shared-slot (uv) part alone +3.2 [2.7, 3.7], 77%. Emergent property gain -0.4
  [-0.6, -0.1], 43% positive (negative for five of six readers, Qwen-14B -0.03); against the
  both-inputs control +0.4 [0.2, 0.6], 56%; the schema alone (inputs not named) -1.7.
- Synergy proxy: fused lines (uv, emergent) do not show more synergy than inherited lines
  (Cliff's delta -0.1 to -0.3, opposite to P3's ordering).
- Judges: generic-space gain is the same whether the panel unanimously accepted g (4.5) or
  rejected it (4.8); emergent-line schema gain for unanimous scope 3 vs 1: 0.45 vs 0.27, delta
  0.05. Coherence: per-token log-probability lower for unanimous incoherent blends (delta -0.28,
  p = 0.02). Surprise vs total gain: Spearman 0.00 to 0.14.
- Reading: the generic space carries information, and it carries it about the shared slot, the
  one property both inputs feed. It does not make the emergent properties more likely than the
  better input alone does; those lines are barely predictable under any input (best single input
  +1.5 over skeleton, both inputs +0.8). P1 and P2 supported; P3 half (inherited lines follow
  their input; synergy ordering not seen); P4 not supported; P6 supported (schema buys +4.4 per
  blend against the analogy mapping's +16.6 per invention).
- Figures: `make_blend_figures.py` -> `downstream/blend_figures/`: two panels, inherited lines
  and emergent lines under input u / input v / both inputs / generic space (colours #D9696B,
  #F0B75B, #488AE5, #9CDC9B), reader-averaged; PDFs copied to the paper's `media/figures/`.
- Example for the figure: Liquid Franchise (sonnet-5, F0) is a weak case (generic space -200.1 vs
  both inputs -199.4 on the block); pick a blend with a clear uv step for the figure.

### Caveat found after scoring (2026-09-19): wording shared with the generic space
- Generators often restate the generic space in a uv line (Gloria: g "devoted attention ...
  transforms the perceiver", uv line "transforms perceiver through devoted attention"). Content
  words shared with g: uv lines 17%, other tags 2-3%. Among uv lines, 258 of 587 share at least
  one content word with g; their schema step is +4.1, versus +1.1 for the 329 with no shared word
  (Spearman between overlap and step 0.43). So roughly half of the uv-line gain is the reader
  reproducing the phrase it was just shown, the same hazard as relation tokens in the old per-fact
  invention measure. The zero-overlap uv step (+1.1) is still the largest of the four tags
  (u -0.3, v +0.2, emergent +0.1), so the schema-to-shared-slot reading survives at a smaller size.
- To do before quoting: report the paper gains on zero-overlap lines as well, or mask shared
  content words in the schema (would need a per-item edit, which the user rejected for controls;
  a subset analysis is the clean alternative).
