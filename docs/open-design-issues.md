# Open design issues

No design question is open. The two below came out of the v2 review. The team decided both on 2026-11-12, and the decisions ship in `prometheon-scoring/3.1`. They stay here as the record of why the rules changed.

## B28: the time budget made model scores depend on the validator's hardware (resolved)

**Decision.** One fixed time limit per model, `[evaluation] model_timeout_seconds` (1,800 seconds), to load and judge its whole corpus. A model that hasn't finished within it gets no score from that validator for the day. It's published as `timed_out` and isn't ranked, and the items it didn't reach count neither for it nor against it. With models capped at 4 billion parameters and a daily corpus of a few hundred to a few thousand posts, a model needs minutes on one 24 GB card, so the limit only stops a model that would hold up the cycle for hours. See [`scoring.md`](./scoring.md#the-time-limit).

This is option 2 below, without a completion floor: a model either finishes or has no score.

**Before, in 3.0.** Each model got `model_timeout_seconds` (3,600 s) of wall-clock time, checked between batches. Once it was spent, every remaining batch was scored as incorrect.

- The same model could score lower on a slower validator, because more of its batches ran past the budget. Validators then disagreed for a reason that had nothing to do with the model, and the slower ones lost vtrust.
- A miner couldn't tell whether a weak score was the model or a validator's hardware.

The options that were considered:

1. Budget in work, not time: cap each model by tokens or forward passes.
2. Exclude unfinished items instead of failing them, and leave a model unranked below a completion floor.
3. Bound the whole day by splitting a fixed budget across the models.
4. Keep it, and rely on the hardware floor.

## B29: the efficiency term measured the tokenizer, not compute (resolved)

**Decision.** The token-efficiency term is gone, with its λ, its P90 normalisation, its spread guard, its token counting, and its config and record fields. Model size takes its place, measured the same way on every validator:

- `P`, the parameter count, is summed from the shapes in the safetensors headers of the model's weight files, read by range request, so nothing is downloaded to count it.
- `P` above 4,000,000,000 makes a model invalid, as too large.
- Past 500,000,000 parameters, a model's accuracy is reduced by `floor(100 × log2(P / 500,000,000))` basis points: one point per doubling.

See [`scoring.md`](./scoring.md#the-parameter-count). This is option 1 below, with the model's size standing for its compute.

**Before, in 3.0.**

```text
t̄ = (policy prefix tokens + content tokens + verdict tokens) / items
score = accuracy × (1 − λ·t)
```

`t` was `t̄` relative to the field's 90th percentile, and λ was 15%. The term switched off when `t̄` varied by less than 10% across the field, which under forced-choice decoding was the usual case. The tokens were counted by the miner's own tokenizer.

- A coarse tokenizer produced fewer tokens for the same text, so it lowered `t` and could gain up to 15% without the model being any cheaper to run.
- One model with an unusual tokenizer could push the spread over 10% and switch the term on for everyone.
- `[evaluation] batch_size` changed how the policy prefix was shared, and so changed `t̄` too.

The options that were considered:

1. Estimate compute from the parameter count.
2. Count tokens with one reference tokenizer only.
3. Drop the term until a real compute measure exists.
4. Keep it.
