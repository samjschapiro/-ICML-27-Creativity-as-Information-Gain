# analogy_ig: progress

## 2026-09-18: track created
- Goal: falsify/validate the five-type taxonomy of analogy information gain
  (`docs/memos/2026-09-18_analogy_information_gain_types.md`) on KOMBINE analogy data.
  Full design in `design.md`.
- Built: `build_dataset.py` (joins comb-creat-eval responses + judge verdicts, all 35 models),
  `score_logprob.py` (types 1 and 3 under a local MLX LM), `render.py` (contexts), `logprob.py`.
- Probability model: Qwen2.5-7B-Instruct 8-bit via mlx-lm, held out from the 35 scored systems.
- Status: dataset built; log-prob scoring smoke-tested in debug mode; full run pending.
- Next: full type-1/3 run, analysis script (P1a-c, P3a-d, dissociability), then phase 2 (types
  4, 2, 5) which needs OpenRouter budget.

## 2026-09-18 (later): Overleaf paper scaffolded
- `papers/6aad5b22035168dccf225738/` now follows the kg_creat-iclr layout: `setup/` (imports,
  configurations, refs.bib, ICML style kit), `sections/00_..14_` numbered in compile order,
  `media/{figures,tables}/`. ICML 2027 kit not yet published; using the 2026 kit, swap later.
- All section bodies are `\ai{}` placeholders pointing at the memo / design doc.
- Full type-1/3 scoring run launched in the background (Qwen2.5-7B-Instruct-8bit, ~2 h).

## 2026-09-18 (evening): phase 1 first run, interim results (Qwen2.5-7B-Instruct-8bit, 956 records)
Run: `data/analogy_ig/dataset/downstream/logprob_qwen7b/` (C_rho used the SOURCE relations; superseded
for the invalid-analogy contrast by `logprob_qwen7b_target_skeleton/`, re-running). Valid-analogy
numbers are unaffected by the fix.

Type 1 (alignment gain), 553 valid analogies:
- P1a supported. Entity-token gain from skeleton to true source path: mean +11.1 nats, 82% positive,
  Wilcoxon p ~ 1e-60. True source beats a donor's entities in 92% of records (mean +17.2 nats,
  p ~ 1e-84). The specific dictionary M carries the information, not "any entities".
- P1b (valid vs relations_differ) confounded in this run; awaiting re-run. Valid vs factual-fail:
  no difference (per-token 0.83 vs 0.80, p = 0.63). Alignment gain is blind to truth, which is what
  the theory says: utility is a separate reference model.
- P1c not supported per token (Spearman -0.06, n.s.); total gain scales with tokens only.

Type 3 (concept-invention gain):
- P3a supported. E_inv higher for integration = True than False among unanimous panels
  (26.6 vs 19.6 nats, n = 242 vs 51, Cliff's delta 0.34, p = 1.5e-4).
- P3b NOT supported. Neither A_B1, A_B0, nor E_inv separates utility = True from False, under
  unanimity (n = 5 negatives) or majority (n = 45 negatives): all p > 0.5, |delta| < 0.07.
  Coherence as judged is not a code-length property of the image triples under this reader.
  Candidate explanations: the utility judge is nearly saturated (508/553 True, ICC 0.55 "fair"),
  or coherence is a property of the concept as a whole rather than of its triples.
- P3c partial. A_B1 tracks integration (delta 0.29, p = 1e-3) but A_B0 does not (p = 0.41):
  the B1 plausibility is contaminated by the mapping information; the clean plausibility (B0)
  carries no integration signal. corr(E_inv, A_B1) = 0.12.
- P3d strongly supported. E_inv exceeds E_src by 27.6 nats on average, 97% of records
  (p ~ 1e-90): known source facts gain nothing from the mapping, invention triples gain a lot.
  Also A_S0 > A_B0 (63%, p = 2e-14): facts are cheap in their domain, definitions expensive.

Dissociability: Spearman(G_M, E_inv) = -0.06, n.s. (n = 553). Types 1 and 3 are uncorrelated.

Next: finish the target-skeleton re-run for P1b; write the phase-1 report; decide on phase 2.

Report: `docs/reports/2026-09-18_analogy_ig_phase1/report.md` (figures + five qualitative examples;
assets from `make_report_assets.py`). P1b section to be filled when the target-skeleton re-run lands.

## 2026-09-18 (later): data loss and re-run
- Rebuilding the dataset with --overwrite (to add the `surprise` field) deleted `downstream/` and
  both scoring runs. See `docs/memos/2026-09-18_overwrite_hazard.md`. `init_directory` now keeps
  `downstream/` on overwrite. Report figures/examples are intact in `docs/reports/`.
- Single re-run launched to `downstream/logprob_qwen7b_target_skeleton/` (target-derived skeleton,
  so it also resolves the mismatched-relations contrast). All downstream configs point there.
- Queued on that run: `analyze_distance.py` (gain vs anchor distance and vs path surprise).

## 2026-09-18 (evening): re-run complete, report finalised
- `logprob_qwen7b_target_skeleton/` (956 records) reproduces every valid-analogy number from the
  lost run exactly (teacher forcing is deterministic).
- Mismatched-relations contrast, now clean: no difference in per-token alignment gain (1.11 vs
  0.83, p = 0.50). Alignment gain is blind to relation identity as well as to factuality; it
  measures entity correspondence only.
- Distance analysis (`analyze_distance.py`): alignment gain falls with path surprise (rho -0.14,
  within-item -0.15, p < 0.002); invention gain rises weakly within item (rho +0.12, p = 0.005);
  anchor distance predicts neither at the item level (n = 30). Figure 5 in the report.
- Family partition (`analyze_family.py`): alignment gain flat across generating families
  (p = 0.19 within item); invention gain differs strongly (p = 3e-6): Phi-4, GLM, Llama families
  well below the item average, Anthropic above. Section + Figure 6 added to the report.
- Multi-reader sensitivity run on Lambda (A100, instance 91948f6e...): 5 held-out readers via a new
  hf backend (`logprob_hf.py`). First launch failed on a transformers 5.x import bug (PIL); pinned
  4.51.3. Second launch failed on pre-existing empty output dirs; removed and relaunched.
- Per-model partition (`analyze_model.py`): invention gain separates the 35 generators (p = 7e-6)
  and correlates with integration pass rate (0.44) and KOMBINE overall (0.39); alignment gain
  correlates with nothing. Figures 7 and 8 in the report. Grok models: perfect integration rate,
  near-zero invention gain; flagged for a manual look.
- Lambda run, third launch: jinja2 3.0.3 on the box broke chat templates; upgraded to 3.1.6.
- Found that the original scoring launch had crashed at record 957/977 on a malformed projection
  triple (glm-4.6, item E9); a pipe in the launch command hid the non-zero exit. Fixed: the scorer
  skips malformed triples explicitly (row carries `skipped`), and a `resume: true` config option
  appends only unscored ids to an existing run. Local Qwen run completed to 977 records; all five
  downstream analyses regenerated; report numbers now on 561 valid analogies. Headline results
  unchanged. Remote runs hit the same crash and will be completed with a resume pass.
- Cross-reader check complete: 5 further readers (Llama-3.1-8B, Mistral-7B v0.3, Gemma-2-9B,
  Qwen2.5-14B, OLMo-2-7B) on Lambda A100, all 977 records each. Every contrast replicates in sign,
  direction and significance; absolute nats vary by reader (invention gain 10 to 36). Per-model
  invention-gain ranking agrees across readers at median Spearman 0.68. Section added to the report.
  Instance terminated. Cost: ~1.5 h of A100.

## 2026-09-18 (night): projection-direction bug and its effect
- Bug: invention contexts assumed the projection always runs u -> v. KOMBINE lets it run either
  way; 139/561 valid analogies run v -> u, 97 undetectable. For v -> u records the "target domain
  only" baseline showed the wrong path. Fixed (`render.projection_direction`, direction recorded
  per row; undetectable cases scored u -> v and flagged `assumed_u_to_v`).
- Effect (Qwen-7B reader, new run `logprob_qwen7b_projdir`): u -> v records identical; v -> u
  invention gain 31.3 -> 23.5 nats (the wrong baseline had inflated it), now equal to u -> v (22.8).
- Record-level conclusions hold: invention > known fact 98%; integration judge delta 0.24
  (p = 0.006, down from 0.34); coherence judge still unseparated; dissociation -0.07; surprise
  correlations similar (invention within-item +0.10, p = 0.02).
- Model-level claims do NOT survive: invention gain vs integration rate 0.11 (was 0.38), vs
  KOMBINE overall 0.05 (was 0.34), vs analogy composite -0.02. Models differ (Kruskal p ~ 0) but
  the ordering no longer tracks benchmark standing; earlier tracking was largely the bug (models
  projecting v -> u more often got inflated gains). Family: Microsoft still clearly lowest (-11);
  Meta -3.8, Zhipu -2.2 (CIs now cross zero); Anthropic +1.2, DeepSeek +2.5, Moonshot +3.4.
- Cross-reader re-run on a fresh A100 (instance 4a93c6f9, 129.213.93.18) into *_projdir dirs.
  A duplicate loop (from an interrupted launch) corrupted the first Gemma file; killed, removed,
  relaunched as a single loop at 20:01 UTC. Report to be rewritten once all readers land.
- Corrected cross-reader run complete (5 readers x 977, `*_projdir`); box auto-fetched and
  terminated by `lambda_autofinish.sh` at 16:04 local. Report rewritten from the invention section
  onward on the corrected runs. Under every reader: invention > fact 98-99%, coherence judge never
  separated, gains uncorrelated, surprise signs consistent, no model-level tracking of KOMBINE.
  Integration judge separated by 5/6 readers (not Gemma-2-9B). Per-model invention-gain rank
  agreement across readers: median Spearman 0.70.
- Total GPU spend today: ~3 h of A100.
- Camera-ready figure set (`make_camera_figures.py`): Nimbus Roman, ICML widths, PDF+PNG, six
  figures showing each quantity per reader plus an across-reader average, and reader-averaged
  per-model / per-family panels. Placed in the paper (Results, captions marked \ai) and the report.
- Histogram version of the invention-vs-known-fact figure (per-analogy values averaged over the six
  readers): invention gain 18.2 nats, known-fact gain -5.2, invention > fact in 99.6% of the 561
  valid analogies. In the paper as Figure 4 and in the report.
- Paper Results pared to what the user wants shown: two reader-averaged histograms (alignment
  gain vs donor: 11.8 vs -3.4 nats, true > donor in 96.4%; invention vs known fact: 18.2 vs -5.2,
  99.6%) and a full-width concrete-examples figure (`make_examples_figure.py`, five records with
  reader-averaged gains and panel verdicts). Forest plots and per-model/family panels removed from
  the paper and from the report's camera section (their numbers stay in the report tables).
- Paper figure text rewritten in plain language at the user's request: no "donor", "skeleton",
  "reader", "forced target"; captions define the measured quantity and carry no commentary;
  example boxes are titled by judge outcome only. Vocabulary now used in the paper: "real source
  path" vs "unrelated source path", "facts about the invented concept" vs "existing facts about
  the source concept", "scored line", "six language models".
- Main-text Figure 1 replaced by a one-example figure (`make_simple_figure.py`): the same scored
  line under the input without and with the analogy, with the log-probability under each (mean
  over six models). Fractional Mandate example: target path -22.2 -> -5.6 (+16.6 nats); invented
  fact -32.4 -> -9.9 (+22.5). Eight-prompt figure moved to the appendix.
- Alignment measure in the paper switched from entity tokens to the whole scored line so the
  histogram, the examples figure and Figure 1 agree: real source path +13.1 nats vs unrelated
  source path -2.2, real higher for 95.9% of analogies (was 11.8 / -3.4 / 96.4% on entity tokens).
  The report's tables keep the entity-token numbers.
- Figure 1 now shows the controls: row (a) baseline (relations only) -22.2, unrelated source path
  -34.2 (-12.0 vs baseline), real source path -5.6 (+16.6); row (b) invented fact -32.4 -> -9.9
  (+22.5) vs an existing source fact -25.0 -> -25.2 (-0.2). The unrelated source path shown is the
  one actually scored (donor RNG sequence replayed from the scoring selection; `score_seed` and
  `score_selection` in the figure config must match the scoring config).

## Decision (2026-09-18): the control for the target-path measurement is the format control
- Considered and rejected: masked-entity source path (reader rebuilds entities from the domain name
  and relations; dropped before scoring), reader-generated source path, shuffled entities, named
  source without a path. Any control that names the source domain lets the reader reconstruct it.
- Kept: the unrelated source path, i.e. a prompt of identical shape with a path from a different
  analogy in the same slot. It isolates the analogy's content from the prompt format. It scores
  below the relations-only baseline (mean -2.2 nats), so the real-path gain is conservative.
- Paper state unchanged (commit 799c7db): baseline / format control / analogy in Figure 1 and the
  alignment histogram; existing-fact control on the invention side.

## Decision revised (2026-09-18, late): named-source control
- Scored two further controls under all six readers as downstream passes of the runs
  (`score_format_control.py`, control: format | named; ~2 min per reader on A100, instance
  auto-terminated by `lambda_autofinish_controls.sh`): (1) content-free path in the analogy format
  (A --r1--> B ...), (2) the source concept named, no path.
- Named source is within 2 nats of the baseline for every reader; real path +9 to +21 above it,
  higher for 84-93% of analogies. Content-free format path is at or below baseline. The paper
  now uses the named-source control (Figure 1, histogram, examples; `control: named` in the three
  figure configs; `format` remains available). Overleaf commit pushed.
- Steelman check (fixed controls, no per-item selection): strongest non-analogy input by mean over
  readers is the relations-only baseline itself (-42.3), then content-free format (-43.0), named
  source (-43.5), unrelated path (-44.5). Real source path over relations-only: +9.4 to +20.4 by
  reader; +13.1 per analogy averaged over readers, higher for 92.5% of 561. This is the alignment
  gain to quote. Figures unchanged (both series are already measured from that baseline).
- Figure 2 redrawn per user: raw log-probability of the target path (reader-averaged) under three
  inputs, colours relational structure #D9696B, analogy instruction #488AE5, analogy content
  #9CDC9B; no legend, x label or annotation. Means -42.3 / -43.5 / -29.2 nats.

## Decision (2026-09-19): invention gain re-measured as a skeleton task over the whole description
- Problem with the per-fact measurement: each invented triple was scored one at a time and about
  half to two thirds of its gain fell on relation tokens copied from the source facts, so the
  number mixed "the mapping helps conceptualise the invention" with "the relation names are
  given". The user asked for the log-probability of the whole description M[Phi] and for a
  framing parallel to the alignment task.
- New measurement (`score_invention_block.py`, attached to each reader run as
  `downstream/invention_block/`): the task fixes the relations, their order, the number of facts
  and the invention's slot ("describe the novel concept '<h>', a concept in the domain of
  '<target>', using these relations in this order, where X, Y, Z stand for concepts you must
  supply"), and the scored block is the whole description M[Phi] rendered as a path. Four
  inputs: structure (task only), instruction (task + "use an analogy with <phi> from the domain of
  <source>"), mapping (task + both aligned paths with correspondences, no source facts), content
  (task + source facts + both paths with correspondences). The mapping input was added at the
  user's request to isolate where the gain comes from. No system message.
- Six readers (five on one A100, ~5 min each, instance auto-terminated; Qwen-7B local). Valid
  analogies, n = 561, log-probability of the whole description averaged over readers:
  structure -44.0, instruction -42.8, mapping -26.2, content -17.0. Consecutive gains: naming the
  source +1.2 (positive for 63%), the aligned paths +16.6 (95% CI [15.5, 17.7], Wilcoxon
  p = 5e-87, positive for 91%), the source facts on top +9.2 ([8.6, 9.8], p = 1e-90, positive for
  97%). Total +27.0 ([25.7, 28.3], positive for 99%). About two thirds of the total arrives with
  the mapping alone (reader range 48% to 74%). Relation tokens contribute nothing in any input
  (mean -0.1 nats), so the gain is on the supplied concepts.
- Per reader, total gain: Qwen-7B 27.7, Llama-8B 19.5, Mistral-7B 30.0, Gemma-9B 24.6,
  Qwen-14B 37.5, OLMo-7B 22.8; the mapping step is the largest step for every reader except
  Gemma (11.5 vs 12.7 for the facts). Integration judge: Cliff's delta 0.25 to 0.44 on the total
  gain, all six readers (was 0.19 to 0.27 per fact). Surprise vs total gain, raw Spearman +0.07 to
  +0.10.
- Figure: `make_camera_figures.py` now draws the invention histogram from these four inputs
  (relational skeleton #D9696B, analogy instruction #488AE5, analogy mapping #F0B75B, analogy
  content #9CDC9B; x range -110 to 0, same as the alignment histogram). Written to a new output
  dir `camera_figures_invblock` (config `make_camera_figures_invblock.yaml`); PDF copied to the
  paper's `media/figures/`. Cross-reader ladder: `analyze_invention_block.py` ->
  `downstream/invention_block_ladder/`.
- Example (Fractional Mandate, three facts, reader-averaged): -53.5 / -45.8 / -22.8 / -10.8.
