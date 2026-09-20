# Session log: 2026-09-18 to 2026-09-19, analogy information gain (phase 1) and paper scaffold

## Summary
Set up the repo and the Overleaf paper, wrote the theory memo on the kinds of information gain an
analogy licenses, built the `analogy_ig` track that estimates two of them (alignment gain, invention
gain) with held-out language models as probability models on the KOMBINE analogy data, validated
the estimates under six readers on Lambda A100s, iterated the control design with the user, and
produced the phase-1 report and the paper figures. Along the way: one data-loss incident (fixed and
guarded), one measurement bug (projection direction; fixed and re-run), and a standing rule that
Claude never uses the overwrite flag.

## Tasks completed
- Read start docs; cloned the Overleaf repo into `papers/6aad5b22035168dccf225738/`.
- Recorded the study framing (KOMBINE x Elmoznino et al. compositionality) in
  `docs/research_context.md`; downloaded arXiv 2410.14817 into `literature/`.
- Wrote `docs/memos/2026-09-18_analogy_information_gain_types.md`: five kinds of gain (alignment,
  candidate inference, concept invention, schema abstraction, restructuring), objects, baselines,
  truth-aptness, reference agent.
- Built the `analogy_ig` track: `build_dataset.py` (1037 analogy heads from 35 models with panel
  verdicts and surprise), `score_logprob.py` (mlx and HF backends; resume mode; malformed-triple
  guard), `render.py` (all prompts), `analyze_logprob.py`, `analyze_distance.py`, `analyze_family.py`,
  `analyze_model.py`, `analyze_models.py` (cross-reader), `make_report_assets.py`,
  `make_camera_figures.py`, `make_simple_figure.py`, `make_examples_figure.py`,
  `score_format_control.py` (format and named-source controls), Lambda runner and unattended
  finishers (`lambda_run.sh`, `lambda_autofinish*.sh`).
- Scored six readers (Qwen2.5-7B local via mlx; Llama-3.1-8B, Mistral-7B, Gemma-2-9B, Qwen2.5-14B,
  OLMo-2-7B on A100) on all 977 selected records, twice (before and after the projection-direction
  fix), plus two extra target-path controls per reader. Total GPU: ~3.5 h A100.
- Phase-1 report `docs/reports/2026-09-18_analogy_ig_phase1/` with figures, qualitative examples,
  cross-reader table, control comparison, family and per-model partitions.
- Paper: standard section names; ICML 2026 kit (2027 not out); Figure 1 is the user's Figma
  overview; all generated LaTeX box figures removed at the user's request; histogram PDFs left in
  `media/figures/` for Figma import. Pushed repeatedly; last commit 159e644.
- Global rule installed: never use the overwrite flag (PreToolUse hook in `~/.claude/settings.json`,
  `~/.claude/CLAUDE.md`, repo `CLAUDE.md`, memory).

## Key results (561 valid analogies, six readers)
- Alignment gain (target path given real source path vs relations only): +13.1 nats per analogy
  averaged over readers, positive for 92.5%. Naming the source concept without its path is worth
  ~0 (within 2 nats of baseline for every reader); content-free and unrelated-path format controls
  sit at or below baseline. Blind to factuality and to relation mismatch.
- Invention gain (invented concept's facts given the full analogy vs target domain only): +18.2
  nats vs -5.2 for existing source facts; invention higher for 99.6%.
- Integration judge separated by 5/6 readers (Cliff's delta 0.19-0.27); coherence judge by none.
- The two gains are uncorrelated (|rho| < 0.1 every reader). Surprise lowers alignment gain and
  slightly raises invention gain, same signs under every reader.
- Model-level gains do not track KOMBINE standing under any reader (earlier correlations were the
  direction bug). Phi-4 and GLM-4.5-air lowest, Gemini-2.5-pro highest on invention gain across
  readers.

## Incidents and decisions
- Data loss: rebuilding the dataset with the overwrite flag deleted `downstream/` runs. Fixed
  `init_directory` to preserve `downstream/`; rule: never use the flag.
- Projection-direction bug: contexts assumed u -> v; 25% of valid analogies project v -> u. Fixed,
  everything re-scored, model-level claims withdrawn.
- Controls: masked-entity path rejected before scoring (reader rebuilds entities); unrelated path,
  content-free format path, named source all scored; named source chosen for the paper; steelman
  = relations-only baseline itself.
- Paper vocabulary: plain terms only, captions define and do not comment.

## Open questions / next steps
- User's question: does an association path between the same two anchors carry information
  about the analogy's target path? KOMBINE's association items use different anchor pairs from the
  analogy items (0 overlap), so this needs new association paths for the 30 analogy pairs.
- Symmetry check: score the source path given the target path (mutual information should be
  roughly symmetric).
- Coherence judge is not visible to any reader at the triple level; decide how the paper states it.
- Grok models: perfect integration pass rate, near-zero invention gain; look by hand.
- Phase 2 (schema transfer, candidate inference, restructuring) designed in
  `docs/tracks/analogy_ig/design.md`, not built.

## Addendum (2026-09-19, evening): invention gain re-measured, alignment p-values
- Re-scrutinised the invention baselines with the user. Old per-fact measure put half to two
  thirds of its gain on relation tokens copied from the source facts. Replaced by a skeleton task
  over the whole description M[Phi] with four inputs (skeleton, instruction, mapping, content);
  the mapping input isolates the locus of the gain. Scored under all six readers
  (`score_invention_block.py`, Lambda instance auto-terminated by `lambda_autofinish_invblock.sh`).
- Result, reader-averaged over 561 valid analogies: -44.0 / -42.8 / -26.2 / -17.0 nats; the
  aligned paths alone are worth +16.6, the source facts a further +9.2, total +27.0 (99% positive);
  about two thirds of the gain arrives with the mapping. Every reader agrees. Recorded in
  `docs/tracks/analogy_ig/progress.md`, the phase-1 report, and the paper's estimator appendix;
  invention histogram redrawn from the four inputs (`camera_figures_invblock`).
- Alignment gain p-values reported: +13.1 nats per analogy, Wilcoxon p = 2e-87; per anchor pair
  (n = 30) all positive, p = 2e-9; per generating model (n = 35) all positive, p = 6e-11.
- Still open: association-vs-analogy test (needs `kombine_retest30`, which shares the 30 pairs
  across tasks but has no judge scores); symmetry check.

## Addendum (2026-09-19, night): blend track built and scored
- Built `blend_ig` as the blending counterpart of the skeleton design; predictions fixed in
  `docs/tracks/blend_ig/design.md` before scoring; the user's framing of joint compression gain and
  emergent property gain adopted verbatim, both-inputs control promoted to a figure panel.
- Scored 1033 blends x 7 inputs under six readers (local Qwen; five on one A100, auto-terminated).
- Result (393 valid blends): generic space +4.4 nats over naming both inputs (80% positive), vacuous
  schema +0.3; the schema's information lands on the uv lines (+2.4 per line, largest step for every
  reader); joint compression gain +2.1 (u, v lines) or +6.0 with uv; emergent property gain -0.4
  (schema does not beat the better single input on emergent lines). Judges invisible. Details in
  `docs/tracks/blend_ig/progress.md`.
