# Types of information gain licensed by analogy

Status: working memo, 2026-09-18. Analogy only; blending deferred.

## Setup (KOMBINE notation)

- Entities C, relations R, triples (c, r, c'). D_u, D_v are the triple sets an
  agent knows about the domains of u and v.
- An analogy selects paths p_u = ((u, r_1, a_1), ..., (a_{n-1}, r_n, a_n)) and
  p_v = ((v, r_1, b_1), ..., (b_{n-1}, r_n, b_n)) sharing the relation
  sequence rho = (r_1, ..., r_n), and induces the structure mapping
  M: a_i -> b_i (with a_0 = u, b_0 = v).
- A projected source element phi with source structure
  Phi = {(phi, r_j, s_j)}_j yields the invention h := M[Phi].
- All K(.) are conditional Kolmogorov complexities relative to a fixed
  reference agent. Which agent matters (see "Reference agent" below).

The single formula from the earlier framing,
EC_an(h) = K(h | D_v) - K(h | D_v, D_u, M), treats "what the target learns from
the source" as one quantity. It is not. An analogy licenses at least five
different gains, about five different objects, measured against five different
baselines. Two of them are not even truth-apt.

## The five types

### 1. Alignment gain (joint compression of the two domains)

**Object:** the pair (D_u, D_v). Nothing new is asserted about either domain.

**What is gained:** the discovery that a subset of D_u and a subset of D_v are
instances of one relational skeleton. Encoding the two paths separately costs
about 2 K(rho) + K(ent_u) + K(ent_v). Encoding them jointly through M costs
K(rho) + K(ent_u) + K(M), where K(M) = K(ent_v | ent_u) is the dictionary.
So

    G_align = K(p_u) + K(p_v) - K(p_u, p_v)  >=  K(rho)  (approx.)

This is algorithmic mutual information between the two paths. It is
retrospective: every triple involved was already known. What changes is the
agent's description of its own knowledge (Schmidhuber's compression progress,
not Shannon information about the world).

**Truth-apt?** Yes, via U_an: the isomorphism either holds (every triple
factual, relation sequence identical) or it does not. If U_an = 0 the joint
code has a nonzero error term and G_align collapses.

**Where the surprise lives:** the dictionary M. Under a semantic-proximity
prior p_sem(b_i | a_i), the alignment is informative to the extent that the
substitutions were improbable: sum_i -log p_sem(b_i | a_i). This is KOMBINE's
S_an. A mapping between near-synonyms has high G_align but was already implied
by similarity, so it teaches nothing. Alignment gain is therefore best stated
net of what the prior already gave away:

    G_align^net = K(rho) - [K(M) - sum_i -log p_sem(b_i | a_i)]

i.e. skeleton reuse minus the part of the dictionary that was predictable.

**Gentner correlate:** highlighting / alignment. **KOMBINE facets:** U_an, S_an.

### 2. Candidate-inference gain (hypotheses about existing target entities)

**Object:** D_v, specifically triples over entities already in the target
domain.

**What is gained:** for any source triple t = (a_i, r, a_j) in D_u whose
endpoints are both in the domain of M but whose image M[t] = (b_i, r, b_j) is
absent from D_v, the analogy licenses M[t] as a hypothesis about the target.
This is Gentner's candidate inference and Holyoak's copy-with-substitution.
The atom/solar-system analogy licensing "electrons orbit the nucleus" is this
type.

    G_infer(t) = K(M[t] | D_v) - K(M[t] | D_v, D_u, M)  =  I(M[t] : D_u | D_v)

The second term is about zero (M[t] is one substitution away from a known
triple). The first term is high iff D_v does not already imply M[t].

**Truth-apt?** Yes, and this is the defining property. M[t] is a claim about
the world that can be checked. The realized gain is therefore conditional on
verification: expected gain = P(M[t] true | world) * G_infer(t). Wrong
inferences (the sun has no analogue of electron spin) cost bits rather than
save them.

**Note on KOMBINE:** the analogy prompt requires the projected element phi to
have NO counterpart in the target. That excludes type 2 by construction. So
the benchmark measures type 3 but not the classical analogical inference.
Worth stating explicitly in the paper, and possibly worth an added task
variant.

**Gentner correlate:** candidate inference / projection.

### 3. Concept-invention gain (new vocabulary in the target)

**Object:** the target domain's language, i.e. its entity vocabulary V_v and
the semantics that gives new symbols meaning (Elmoznino's W and f).

**What is gained:** phi in D_u has no image under M. The analogy extends the
mapping to M' = M + {phi -> h} where h is a fresh symbol whose meaning is
*defined* by its inherited structure h := M[Phi]. "Accruing franchise" is not
a hypothesis that something exists; it is a definition of a slot the target
domain did not have.

The cost side is what makes this creative rather than arbitrary:

    K(h | Phi, M) ~ 0      (the definition is one substitution away)
    K(h | D_v) is large    (the target could not have produced it)

But the benefit is prospective, not realized at invention time. A definition
pays off only if future target-domain data D_v' compresses better with it:

    G_invent = K(D_v' | D_v) - K(D_v' | D_v, h)

At the moment of invention this is unknown. What can be measured now is
(a) faithfulness, K(h | Phi, M) ~ 0 (KOMBINE's J^qua_an), (b) coherence, i.e.
h has nonzero probability under a world model of the target (J^utl_an), and
(c) originality, -log p_pop(h) (O_an(h)).

**Truth-apt?** No. h is a definition and cannot be false, only incoherent,
unfaithful to the mapping, or useless. This is why KOMBINE's utility judge for
inventions asks about coherence, not fact. Conflating type 2 and type 3 under
one EC formula misreads a definition as a claim.

**Gentner correlate:** none precisely; closest is Fauconnier and Turner's
emergent structure, which is why analogy and blending overlap here.

### 4. Schema-abstraction gain (a reusable program)

**Object:** the space of domains, not u or v.

**What is gained:** stripping the entities from the aligned paths leaves a
schema S = rho with typed role slots (Gick and Holyoak's schema induction).
S is a small program: given a new domain w, S plus a dictionary reproduces a
path in D_w.

    G_schema(w) = K(D_w) - K(D_w | S)     for unseen w

This is second-order: the analogy between u and v teaches something about
every domain that instantiates S. It is exactly Elmoznino's modular f reused
across inputs, and it is what a blend's generic space g makes explicit. An
analogy with high G_schema is one whose skeleton transfers; one with low
G_schema is a one-off coincidence between two domains.

**Truth-apt?** Indirectly: S is validated by whether third domains fit it.

**Note on KOMBINE:** not measured at all, since each item is one pair. A cheap
test would be to take the model's rho from item (u, v) and ask whether it
instantiates on a held-out third entity w.

**Gentner correlate:** schema abstraction / progressive alignment.

### 5. Restructuring gain (a shift in the target's prior)

**Object:** the prior p_w over descriptions of the target domain, i.e.
Elmoznino's K(p_w) term, not the sentences W.

**What is gained:** after the analogy, the target is described around the
source's relational structure. No new triple need be asserted; what changes is
which existing target relations are treated as central, which is a change in
the distribution over target descriptions.

    G_restruct = KL( p_after(. | D_v, M) || p_before(. | D_v) )

This is Bayesian surprise (Itti and Baldi) over the target's own description
language, and it is the closest thing to Gentner's re-representation and
restructuring.

**Truth-apt?** No. It is a change of representation, and its value is again
prospective (does the re-described target compress better later).

**Note:** this is the least crisp of the five and could be folded into type 1
as "alignment gain seen from the target's side". I would keep it separate
because types 1 and 5 have different objects (the pair vs the target's prior)
and different signs of surprise (1 is zero world-surprise; 5 is nonzero
surprise about the target's own organization).

## Summary table

| Type | Object gained about | Quantity | Truth-apt | Realized when | KOMBINE facet |
|---|---|---|---|---|---|
| 1 Alignment | the pair (D_u, D_v) | I(p_u : p_v) ~ K(rho), net of p_sem | yes (U_an) | now | U_an, S_an |
| 2 Candidate inference | existing target entities | I(M[t] : D_u \| D_v) | yes, checkable | on verification | excluded by prompt |
| 3 Concept invention | target vocabulary V_v | K(h\|D_v) high, K(h\|Phi,M) ~ 0; payoff K(D_v'\|D_v) - K(D_v'\|D_v,h) | no (coherence-apt) | prospective | J^qua, J^utl, O(h) |
| 4 Schema abstraction | space of domains | K(D_w) - K(D_w \| S) | indirectly | on transfer | not measured |
| 5 Restructuring | target's prior p_w | KL(p_after \|\| p_before) | no | prospective | not measured |

## Reference agent

Every K above is relative to an agent. Three are in play and they give
different answers:

- **The maker.** Already knows D_u and D_v. Gets types 1, 4, 5 for free at
  discovery time, and types 2, 3 as hypotheses/definitions.
- **A reader who knows D_v but not D_u.** For them the analogy is a
  transmission code: K(p_v | p_u, M) = K(M) bits to learn the target path.
  This is the pedagogical use of analogy, and it is a sixth thing that is not
  creativity at all (the facts are old; only the reader is new).
- **The population / world model.** KOMBINE's originality uses the response
  pool; utility uses a factuality judge. Neither is the maker.

The paper should fix the reference agent once (I suggest: the maker, with the
world model as the truth oracle for type 2 and the population as the prior for
originality) and say so, otherwise the formulas silently switch agents.

## What this buys the paper

- Finding 2/3 (blending harder than analogy) restates cleanly: analogy tasks
  as posed only require types 1 and 3; blending additionally requires type 4
  made explicit (the generic space is the schema) before anything else can be
  scored. Abstraction is the bottleneck because type 4 is program induction.
- The tension with Elmoznino's disentanglement preference lives in type 3
  versus type 4: type 4 wants a simple reusable f; type 3 wants f rich enough
  that projecting one new symbol yields structure the target lacked.
- KOMBINE currently scores 1 and 3 and omits 2, 4, 5. That is a concrete,
  honest statement of what the benchmark measures.

## Open questions

- Is type 5 genuinely distinct or a change of viewpoint on type 1?
- For type 3, is there a computable proxy for prospective payoff, e.g. does
  conditioning on h raise log-prob of held-out target-domain text?
- Should type 2 be added to the benchmark as a separate task?
