# Session log: 2026-09-26, reuse track

## Summary
User's thesis: once a structure mapping is found it can be reused on many tasks. Built `reuse_ig`
with two tests (held-out fact; reuse across items and across tasks), predictions fixed in
`docs/tracks/reuse_ig/design.md`, scored under six readers (local Qwen; five on one A100,
auto-terminated), results in `docs/tracks/reuse_ig/progress.md`.

## Results in one line each (per item, averaged over readers)
- Held-out target hop: source path +3.2 nats (76%); the mapping licenses a fact it never showed.
- Held-out projected fact: mapping +3.3 (69%), source facts +4.0 more.
- Within-task demonstrations: +1.6 for one, +4.8 for three (relations-only); a third of the
  item's own source path at best; reader-dependent.
- Analogy into blend, same pair: +3.1 (72%); other pair -0.6; stacks with the generic space.
- Blend's generic space into analogy: -2.1 without the source path, +1.7 with it.

## Earlier in the session (2026-09-20 to 21)
Kolmogorov section made sound and reordered; Kepler example (Gentner 2002) added; blend track
scored; invention figures (grey skeleton, gain line with 95% CI); Moral refresher example.
