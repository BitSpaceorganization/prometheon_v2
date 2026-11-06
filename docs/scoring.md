# Scoring

How one day becomes one weight vector, under `scoring_version = "prometheon-scoring/3.0"`.

Every number on this page is in `[scoring]` in your config, and the example configs set each one explicitly. They're not enforced by consensus. Two validators running the same code with different values compute different vectors from the same inputs, and the odd one out loses vtrust. A config without `scoring_version`, or with a different one, refuses to load, so a copy made under older rules fails loudly instead of running new code on old numbers.

All arithmetic is in exact integers and fractions. No float touches a payout.

```text
1,000,000 units per validator per day
├─ 700,000  data pool      top 10 eligible miners, in proportion to data score
└─ 300,000  model pool     250,000 / 40,000 / 10,000 to the top three models
anything nobody qualifies for → the subnet owner hotkey (burn)
```

The split starts at 70% data and 30% models, because real contributors cost far more today than training a text model. The team expects the model and agent share to reach 80 to 90% as the user base grows and the model requirements get harder. That will ship as a new scoring version.

---

## Who is scored

The day's **signed epoch** lists every neuron registered at the freeze, with two counts from BitFan. A miner is **eligible** when:

```text
verified_active_members ≥ min_verified_active_members   (10)
members                 ≥ min_members                   (30)
```

Every member is phone-verified, and `members` includes the verified active ones. One gate covers both pools. An ineligible miner's test posts still enter the corpus, because they're verified human content, but the miner earns nothing, and its model isn't evaluated.

The hub also publishes its own `eligible` flag, for its dashboard. Validators don't use it. They apply their configured floors to the two counts.

---

## Data pool, 70%

### 1. Near-duplicates out

Before anything is labelled, the whole day's test corpus is de-duplicated across all miners. Each post's text is normalised:

1. Unicode NFKC
2. lower case
3. every run of whitespace becomes one space
4. punctuation, symbols, format and control characters, and variation selectors are deleted
5. runs of spaces collapse to one, and the ends are trimmed

Two posts are the same post if their normalised texts are identical, or if their 64-bit SimHashes, over character 3-grams, are within `near_duplicate_max_hamming` bits (3). Posts are compared in `(created_at, id)` order. Each is checked only against the posts kept so far, and only the earliest of a group is kept, whoever wrote it.

### 2. The score

For each eligible miner's remaining posts:

```text
S      = posts remaining after near-duplicates
V_pos  = claimed violating, and the labeller says violating
V_neg  = claimed compliant, and the labeller says compliant

amount  = 2 × min(V_pos, V_neg)
quality = (V_pos + V_neg) / S
score   = amount × quality
```

The labeller is the validator's own, prompted with [`content_policy.md`](../content_policy.md). A post it excludes as unusable counts in `S` and in neither `V`.

| S | V_pos | V_neg | amount | quality | score |
|---|---|---|---|---|---|
| 10 | 5 | 5 | 10 | 1 | **10** |
| 10 | 6 | 4 | 8 | 1 | **8** |
| 10 | 3 | 5 | 6 | 0.8 | **4.8** |
| 40 | 40 | 0 | 0 | 1 | **0**, a flood of one class |
| 20 | 5 | 5 | 10 | 0.5 | **5**, half the posts were wrong |
| 0 | 0 | 0 | 0 | | **0** |

### 3. The split

The top `dataset_top_n` (10) miners with a score above zero split 700,000 units in proportion to score. If fewer qualify, those present share the whole pool. Ranking is score descending, then hotkey ascending. If nobody scores above zero, the pool burns.

---

## Model pool, 30%

### Which models run

A model is evaluated only if all of these hold:

- it's in the epoch's frozen `models` list with `payment = "paid"`
- its hotkey is eligible
- it passes the registry checks against Hugging Face: manifest, architecture, size, and the weights fingerprint must equal the one the hub froze
- no earlier model has the same fingerprint, where earlier means first commit block, then hotkey

The others are published with a status: `unpaid`, `duplicate`, `ineligible`, `invalid` or `unreachable`.

### Accuracy

Each model runs over the day's labelled corpus: every test post that survived de-duplication, from all miners, plus the production sample. **Test posts from the model's own coldkey are left out**, across every hotkey under that coldkey.

```text
accuracy = correct / scored
```

A batch that fails is scored incorrect. If every batch of a model fails, the validator checks that it can still run a model at all. If it can't, the cycle stops instead of scoring the field zero.

### Efficiency

```text
t̄ᵢ    = total tokens ÷ items whose batch reported usage
P₉₀   = 90th percentile of t̄ across models that reported usage
tᵢ    = clamp(t̄ᵢ / P₉₀, 0, 1)
score = accuracy × (1 − λ·tᵢ)                 λ = efficiency_lambda_bp = 15%
```

- **The spread guard.** If the coefficient of variation of `t̄` across the field is below `efficiency_min_cv_bp` (10%), λ is zero for the day. Forced-choice decoding makes token counts nearly uniform, so this is the usual case.
- **A model that reported no usage isn't free.** It takes `t = 1` and stays out of `P₉₀` and the spread guard.
- **Multiplicative, and capped at 15%.** Efficiency can't overturn a wider accuracy gap.

### Ranks and places

- A model scored on fewer than `model_min_scored_items` (100) items isn't ranked.
- A score of zero is never paid.
- Ranking is score descending, then accuracy descending, then hotkey ascending.

```text
model_rank_shares_bp = [2500, 400, 100]      shares of TOTAL emission

three winners   25%   4%   1%
two winners     25%   5%           the empty third place goes to the lowest winner present
one winner      30%
none            the 30% burns
```

---

## Burn

Burned units are weighted to the subnet owner hotkey, read from the chain.

| Situation | Burns |
|---|---|
| No eligible miner scores above zero for data | 700,000 |
| No model qualifies | 300,000 |
| Both | everything |
| A paid hotkey deregisters between scoring and sending | that hotkey's units |

A departed hotkey's units are never redistributed to the rest of the field. That would let a single deregistration change everyone else's pay.

---

## From units to the chain

Units resolve to uids against a fresh metagraph at send time, then scale to `u16` values summing to exactly 65,535. The scaling uses largest remainder, with ties broken by remainder, then units, then hotkey, then uid. The SDK then rescales so the largest weight is 65,535, which is what the chain stores.

Every allocation on this page uses largest remainder too. Floors first, then one unit at a time to the largest remainders, with ties by hotkey. Parts always sum to the pool exactly.

### The subnet's weight limits

The chain refuses a vector that names fewer uids than `min(subnet size, min_allowed_weights)`, or in which one uid's share exceeds `max_weight_limit / 65535`. The only exception is a single weight on the validator's own uid. Validators read both limits before every send and bend the vector to them, the same way on every validator:

- **Too few uids.** The lowest-numbered registered uids not already in the vector are added, never the validator's own unless nothing else is left. Each gets `ceil(1,000,000 / 65535) = 16` units, the least the chain carries, taken from the largest entry.
- **A share over the cap.** It's cut to the cap. The excess is emission nobody may be paid, so it burns to the owner hotkey, up to the same cap. Only what the burn can't hold, when the burn is itself over the cap, is shared among the other entries in proportion to what they hold.
- **Too few entries to fit the cap at all**, for example everything burning with a 50% cap and a validator that isn't the owner. The validator refuses to send rather than pay uids nobody scored.

On the defaults a new subnet gets (`min_allowed_weights = 1`, `max_weight_limit = 65535`), nothing changes. **The owner should keep it that way.** The rules can burn up to 100% of a day, and a cap below 65535 makes such a day unsendable for every validator but the owner. The published record carries the vector as sent, after shaping, so a mirror reproduces it exactly.

## Worked example

These are the numbers in `tests/integration/test_validator_run.py`, which drives `prometheon validator run` end to end and checks each one. The setup:

- **Miners:** A, B, C and D. D has 14 verified active members but only 29 members, so it's ineligible.
- **Data:** A has 6 violating and 4 compliant posts, all confirmed. B has 5 and 5. C has 3 confirmed violating and 7 claimed compliant, of which the labeller finds 2 violating.
- **Models:** A misses nothing, C misses 3 items and B misses 6.

```text
data scores        A 8      B 10     C 4.8         total 22.8
exact shares       A 245,614.04   B 307,017.54   C 147,368.42
floors             245,614 + 307,017 + 147,368 = 699,999 → 1 left, largest remainder B
data units         A 245,614      B 307,018      C 147,368

models             A first 250,000    C second 40,000    B third 10,000

units              A 495,614      B 317,018      C 187,368      burn 0
u16                A 32,480.06    B 20,775.77    C 12,279.16  → floors 65,534, +1 to B
sent               A 32,480       B 20,776       C 12,279
```

Now make every eligible miner claim only violating posts. Each has `V_neg = 0`, every data score is 0, and the 700,000 burns:

```text
units              owner 700,000   A 250,000   C 40,000   B 10,000
u16                45,874.5        16,383.75   2,621.4    655.35   → floors 65,533
sent               45,875          16,384      2,621      655      (+1 to A, +1 to owner)
```

## The published record

After sending, each validator signs a record of the day and posts it to the hub. For every eligible miner, and every hotkey with a listed model, the record carries:

- `S`, `V_pos` and `V_neg`, and the data score in millionths
- the model status, accuracy, rank and score
- `weight`, the units that hotkey earned

`burned_weight` is everything sent to the owner hotkey. The rows plus `burned_weight` sum to exactly 1,000,000 and reproduce the vector the validator sent. The record also names the epoch it scored, by `epoch_content_hash`, so anyone holding the same epoch can recompute it.

## Determinism, and where it ends

Given the same epoch, the same labels and the same model verdicts, every validator computes the same vector, with no clock, no randomness and no dependence on iteration order. Two inputs can legitimately differ between validators:

- **Labels.** The labeller is a language model, and two validators can disagree on a borderline post.
- **Model verdicts on near-ties.** Different GPUs can put slightly different probability on `YES` and `NO`.

Chain consensus arbitrates both. That's what it's for.
