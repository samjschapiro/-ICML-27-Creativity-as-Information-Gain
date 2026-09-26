# reuse_ig progress

## 2026-09-26: built and scored under six readers

Items: 2,339 (378 held-out hops, 503 held-out facts, 508 within-task, 561 blend-to-analogy,
389 analogy-to-blend), 8,028 contexts per reader. Qwen-7B local (17 min); five readers on one
A100 (about 8 min each), instance auto-terminated. Analysis: `analyze_reuse.py` ->
`data/reuse_ig/items/downstream/reuse_analysis/`. All gains below are per item averaged over the
six readers, with 95% bootstrap CI, share positive, Wilcoxon p.

### Test 1: held-out fact
- 1a held-out target hop (378): source path over target-so-far +3.2 [2.7, 3.7], 76%, p = 1e-31;
  positive for every reader (Llama +1.1 at 53% the weakest, Qwen-14B +5.2 at 78% the strongest).
  The bare instruction "by analogy with u" costs 2.2 nats (14% positive). Leak check: the held-out
  entity appears verbatim in the source-path prompt for 2 of 378 items. P1 supported.
- 1b held-out projected fact (503): mapping over known facts +3.3 [2.8, 3.8], 69%, p = 2e-29;
  source facts a further +4.0 [3.6, 4.3], 89%. Instruction alone -0.7. P2 supported.

### Test 2: reuse across items and tasks
- 2a within-task (508, 27 pairs): one fixed demonstration +1.6 [1.1, 2.1] (60%) on the
  relations-only input and +1.1 (63%) with the source path; three demonstrations +4.8 [4.3, 5.2]
  (84%) and +2.4 (79%). Dose-response is real (three minus one: +3.1, 88%). Strongly
  reader-dependent: Mistral +16.8 / +18.5, Qwen-7B -5.8 / -0.6, the rest within +/-4. Three
  demonstrations recover 36% of what the item's own source path is worth (+13.3). Reading: a
  demonstration teaches the operation and the format, a little; it does not substitute for the
  mapping. P3 as predicted (small, less than the source path).
- 2b analogy to blend (389): the pair's own analogy +3.1 [2.6, 3.7] (72%, p = 3e-25) over being
  told the blend's inputs; an analogy between two other concepts -0.6 (42%); paired difference
  same minus other +3.8 [3.3, 4.3], 77%, p = 2e-36, and same beats other for every reader. The
  analogy stacks with the generic space: +3.7 on top of the analogy, +2.4 on top of the generic
  space. Wording check: gain is +3.6 for the 198 blends sharing no content word with the analogy
  and +4.0 for the 191 that share some, so it is not copying. P4 supported in this direction.
- 2b blend to analogy (561): the pair's generic space lowers the target path's log-probability
  when only the relations are given (-2.1 [-2.6, -1.7], 30%), more than another pair's schema
  does (-0.8); once the source path is shown it adds +1.7 [1.4, 2.1] (68%). Reader-dependent
  (Qwen-7B -9.4 dominates the relations-only number; Llama +1.1). P4 not supported in this
  direction: a schema sentence does not help write the path.

### Reading for the thesis
Reuse holds when what is reused is the mapping itself: it licenses facts it never displayed
(1a, 1b) and it lowers the cost of a different task on the same concepts (analogy into blend),
specifically and on top of that task's own structure. Reuse is weak or absent when it would have
to be abstract: a demonstration on other concepts is worth a third of the item's own mapping at
best, and a schema sentence is worth nothing without the path. Consistent with the analogy and
blending results: the information is in the aligned entities, not in the description of the
relation between domains.
