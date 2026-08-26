# Scoring

How a day's work becomes one weight vector. Every constant here lives in
`[scoring]` in your config, so a validator can read the numbers that decide
emissions without reading the code.

A fixed share of miner emission (`miner_burn_share_bp`, **40%** by default)
burns to the subnet owner (uid 0) before any miner is paid. Miners compete for
the remaining **60%** — the miner pool — which splits two to one:

```text
100% miner emission
├─ 40%  burn                    — withheld to the owner (uid 0) every day
└─ 60%  miner pool
   ├─ ⅔  dataset contribution  — did you supply usable evaluation data?
   └─ ⅓  model performance     — is your model the best judge of it?
```

So dataset contribution is **40%** of total emission and model performance
**20%**. `dataset_share_bp = 6667` is that two-to-one split expressed in basis
points of the pool; it lands at 40.002%/19.998%, since an exact two thirds is
not representable there.

Data is weighted the more heavily of the two deliberately: it is what every
model is measured against, it accrues daily from real users, and it is the half
a miner can earn without a deployed model at all.
All of it is computed inside one cycle and combined into a **single**
`set_weights`; the chain sees one final weight vector, burn included.

---

## Dataset contribution (⅔ of the pool, 40% of total)

A miner's users submit balanced test content: for each item they claim whether
it violates policy or is compliant. Each validator labels every item with its
own labeller and rewards only the claims the labeller confirms — so the claim is
a prior, never trusted as ground truth, and a member cannot inflate `V` by
claiming everything violates.

```text
V = items correctly labelled      (the claim matched the labeller's verdict)
S = items submitted
score = V × (V / S)  =  V² / S
```

The second factor is submission accuracy, and multiplying by it is what makes
noise unprofitable:

| Submitted | Valid | Accuracy | Score |
|---|---|---|---|
| 100 | 50 | 50% | **25** |
| 50 | 50 | 100% | **50** |
| 200 | 100 | 50% | **50** |

Submitting fifty good items beats submitting a hundred items of which half are
junk, by exactly the factor by which the junk diluted them. Volume still pays,
but only when it is accurate volume.

`S = 0` scores `0`. There is no division by zero, and a miner who submitted
nothing contributed nothing.

**The pool goes to the top 10 contributors**, split proportionally to score.
Fewer than ten eligible contributors and the pool divides among those present
rather than burning the remainder.

---

## Model performance (⅓ of the pool, 20% of total)

### Accuracy

Every eligible model is run over the whole corpus: test content from *other*
miners carrying whichever verdict labelling gave it, plus production content. It
is scored on how often it agreed with the ground truth (the validator's own
labeller).

The test corpus is now deliberately balanced — submitters label roughly half
their items compliant — so a model can no longer win by answering "violating" to
everything: accuracy only rises by getting both classes right. A correctly
labelled compliant item is a genuine negative and one of the harder ones to
identify; a mislabelled item still costs its author (it counts in `S`, not `V`)
and is measured on the model half too.

```text
accuracy = correct / total
```

Raw accuracy, for now. It is prevalence-sensitive: as the number
of eligible miners grows, test positives accumulate and the corpus tilts toward
`YES`, which flatters a model that simply answers `YES`. Admitting non-violating
test content slows that drift -- it is the only negative supply that grows with
the field rather than independently of it -- but does not stop it.

That is survivable while the field is small and production content dominates,
and it stops being survivable later. **The trigger to watch is the ratio of
test positives to production negatives.** Once positives exceed roughly nine
times negatives, a degenerate always-`YES` model starts beating a competent
one, and accuracy should move to balanced accuracy. Validators log the ratio
every cycle so the switch is made on evidence rather than on a hunch.

### Efficiency

Lower token usage is better, but only slightly. Accuracy is what matters.

```text
t̄ᵢ    = total tokens ÷ items whose batch reported usage
T₉₀   = 90th percentile of t̄ across models that reported usage
tᵢ    = clamp(t̄ᵢ / T₉₀, 0, 1)
score = accuracy × (1 − λ·tᵢ)                  λ = 0.15
```

The denominator is **items whose batch reported usage**, not items scored. A
failed batch is scored incorrect but reports no tokens, and averaging its zero
in would make a broken model look cheap. The two differ for any model that had
a batch fail.

A model that reported no usage **at all** is not treated as free. It takes
`t = 1`, the full discount, and takes no part in `T₉₀` or the spread guard.
Otherwise a deployment that answered nothing would be the most efficient model
in the field, and every model would have an incentive to stop reporting.

Five choices behind that formula:

**Per-item mean.** Self-exclusion means each miner is scored on a slightly
different set, so totals are not comparable across miners.

**The 90th percentile.** Taking the maximum instead would let one model with a
runaway prompt inflate the denominator and squash everyone else's differences
into nothing.

**Multiplicative, not additive.** A cheap-but-wrong model cannot buy rank:
`0.40 × 1.00` still loses to `0.60 × 0.85`.

**λ = 0.15 caps it.** Efficiency can never overturn an accuracy gap wider than
fifteen points.

**The guard.** Decoding is forced-choice, so completion tokens are one per item
for everyone and the real variation is tokenizer efficiency on the policy and
content, a narrow spread. When the coefficient of variation across models falls
below 10%, λ is set to **zero** for that cycle. A term that is not
discriminating must not contribute randomness to close rankings.

### Rank shares

```text
rank 1 → 30%   of total miner emission
rank 2 → 15%
rank 3 →  5%
             = the model third of the pool
```

Ranking is by score descending, with ties broken on accuracy descending and
then hotkey ascending. Never on dictionary order, which would make the result
depend on iteration and differ between validators.

With fewer than three valid models the shares **renormalise** across those
present. Two models take `3000/4500` and `1500/4500` of the model pool, so
nothing is left unallocated because the field was thin.

Two gates decide whether a model may hold a rank share at all:

**It must have been measured.** A model scored on fewer than
`model_min_scored_items` items (default 100) is not ranked. Accuracy over one
item is 100% or 0% and means nothing either way, but it would sort above a
model that answered 4999 of 5000 correctly and take twice its emission. This is
reachable rather than hypothetical, because a miner's own test content is
excluded from the corpus it is scored on.

**It must have scored above zero.** Ranking in the top three is not the same as
having earned something. A model that answered nothing all day scores zero, and
a zero is never paid. Otherwise a dead deployment collects, and a field in
which *every* model scores zero would distribute half of miner emission in
hotkey alphabetical order, identically on every validator, where consensus
could not correct it.

---

## Burn

The subnet owner's hotkey receives any portion with nobody eligible to claim
it. There is no standing burn in normal operation. With eligible miners present,
the full miner emission is distributed.

| Situation | Result |
|---|---|
| No dataset-eligible miners | the dataset two thirds burns |
| No model-eligible miners | the model third burns |
| No model scored above zero | the model third burns |
| No model met the measurement floor | the model third burns |
| Neither half has a claimant | the whole miner emission burns |

**Everything above happens behind one entry gate.** A hotkey is scored at all
only if it is on the day's eligible-miner list, which the subnet data layer
derives from Fan Group active-member counts. Not on the list, and neither half
pays: no commitment is read, no model is evaluated, and any test content its
users wrote is not counted towards the dataset half.

Past that gate the two streams are independent. A miner whose users contributed
that day but who has deployed no model earns from the dataset pool; a miner with
a deployed model whose users contributed nothing earns from the model pool.

---

## Allocation

Both pools allocate integer weight units by **largest remainder**, so the parts
sum to the pool exactly rather than approximately. Remainder ties break on
hotkey ascending.

Floating-point shares that "nearly" sum to the pool produce a different final
vector on different machines, and two validators that disagree by one weight
unit have diverged for no reason at all.

---

## Determinism, and where it ends

Everything on this page is a pure function of the labelled corpus and the
evaluation results. Given identical inputs, every validator computes an
identical weight vector. No clock, no randomness, no iteration order.

The inputs are not identical, and that is expected. Ground-truth labelling
calls a language model, and two validators labelling the same borderline item
can legitimately disagree. Chain consensus arbitrates the difference; a
validator whose labels drift far from the field loses vtrust.

Model inference is a smaller source of divergence than it appears: forced-choice
decoding is deterministic given weights and input, so two validators sending the
same item to the same deployment get the same verdict, floating-point boundary
cases aside.
