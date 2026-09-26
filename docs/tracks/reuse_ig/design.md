# reuse_ig: is a structure mapping reusable?

Thesis (user, 2026-09-26): once a structure mapping is found, it can be reused on a wide range of
tasks. In the paper's terms, a mapping M is a program whose cost is paid once; every task whose
answer lies in the systematic closure of M gets its gain for free. Two tests, fixed before scoring.

## Test 1: held-out fact

The mapping is shown, the fact to be scored is not. Does the mapping license inference of a fact
that was never displayed? (Gentner's candidate inference.)

**1a. Held-out target hop.** Valid analogies with at least 3 hops (378). Scored line: the last
triple of the target path, `s_{n-1} --r_n--> s_n`. The reader always sees the first n-1 target
triples and is told the last relation. Inputs:

| input | what is added |
|---|---|
| target so far | nothing more |
| analogy instruction | "Do so by analogy with '<u>'." |
| source path | the full source path, with the correspondences for the n-1 known entities |

The held-out entity s_n is the counterpart of c_n, which is on the source path but has no stated
correspondence. Scored under the same system message as the alignment measurement.

**1b. Held-out projected fact.** Valid analogies with at least 2 projected facts (503). Scored
line: the last image fact `h --r_m--> M(c_m)`. The reader always sees the invention's other facts
and the relation of the missing one. Inputs: nothing more / analogy instruction / mapping (both
paths with correspondences) / mapping plus the source facts. Same format as the invention task,
no system message.

## Test 2: reuse across items and across tasks

**2a. Within-task reuse.** Does seeing an analogy between X and Y make the analogy between A and
B easier to find? Valid analogies from 27 pairs (the three pairs used as demonstrations are
excluded). Scored: the whole target path, as in the alignment measurement. Inputs: the existing
relations-only and source-path inputs, each with and without a fixed demonstration analogy
prepended (one demonstration; and three). Demonstrations are fixed for every item: Vaccines and
Ethics (kimi-k2), Christianity and Beauty (claude-opus-4.8), The blue whale and The mattress
(gpt-5-mini), all with unanimous judges.

**2b. Cross-task reuse, same pair.** Analogy and blend items share the 30 pairs.
- Analogy to blend (393 valid blends, minus the pair used as the other-pair demonstration): the
  blend's whole description under the skeleton task, with inputs: told it blends u and v
  (existing) / plus the analogy between u and v / plus the generic space (existing) / plus both /
  plus an analogy between two other concepts (Vaccines and Ethics, fixed). The same-pair analogy
  is the strict-valid analogy for the pair from the alphabetically first model other than the
  blend's own.
- Blend to analogy (561 valid analogies): the target path with inputs: relations only (existing)
  / plus the pair's generic space / source path (existing) / source path plus generic space /
  relations plus the generic space of another pair (Vaccines and Ethics blend, fixed). The
  same-pair generic space is from the valid blend of the pair by the alphabetically first model
  other than the analogy's own.

## Predictions

- P1 (held-out hop): the source path raises the log-probability of the held-out target triple
  over the target-so-far input for most analogies, and by more than the instruction does.
  Falsified if a mapping only helps facts it displays.
- P2 (held-out projected fact): the mapping alone raises the held-out fact's log-probability;
  the source facts add more, since they contain the counterpart of the held-out fact.
- P3 (within-task reuse): a demonstration lowers the cost of the target path by a small amount
  relative to the source path itself (operation learned, not content), and three demonstrations
  by at most a little more. Falsified in the interesting direction if the demonstration is worth
  as much as the source path; falsified in the other if it is worth nothing even with the source
  path shown.
- P4 (cross-task, same pair): the analogy lowers the blend's cost beyond naming the inputs, on
  the inherited lines especially, and the pair's generic space lowers the target path's cost;
  an analogy or generic space for another pair does not. Falsified if only the same-task
  structure carries information.

## Readers and compute

Same six readers. About 14 extra scorings per analogy and 4 per blend: roughly 9,000 short
passes per reader.

## Caveats

- Held-out here means held out of the context, not of the generator's distribution: every
  scored line was produced by the same system that produced the mapping.
- Demonstrations and other-pair inputs are fixed choices, not per-item; the three demonstration
  pairs are excluded from the items they could leak into.
- Wording shared between an input and the scored line is checked, as for the generic space.
