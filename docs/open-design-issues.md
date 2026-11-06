# Open design issues

Two findings from the v2 review are design questions, not bugs with one right fix. They're written up here for the team to decide. Until it does, the code behaves as described under "Today". Either change would ship as a new scoring version that every validator adopts together.

## B28: the time budget makes model scores depend on the validator's hardware

**Today.** Each model gets `[evaluation] model_timeout_seconds` (3,600 s by default) of wall-clock time. The clock is checked between batches. Once it's spent, every remaining batch is scored as incorrect. A single slow batch isn't interrupted. A day's total time has no bound: up to one budget per evaluated model, plus downloads.

**Why it matters.**

- The same model can score lower on a slower validator, because more of its batches run past the budget. Validators then disagree for a reason that has nothing to do with the model, and the slower ones lose vtrust.
- A miner can't tell whether a weak score was the model or a validator's hardware.
- The hardware floor in [`validator.md`](./validator.md#hardware) keeps this rare, but nothing in the code compensates when it happens.

**Options.**

1. **Budget in work, not time.** Cap each model by tokens processed or forward passes, which are the same on every machine. A model that exceeds the cap is too large for the subnet, on every validator alike.
2. **Exclude unfinished items instead of failing them**, and leave a model unranked below a completion floor, say 95% of its corpus. Accuracy stops depending on speed. A model too slow to finish anywhere earns nothing, rather than a partial score.
3. **Bound the whole day.** Split a fixed daily budget across the evaluated models, so a cycle always finishes before the next one starts.
4. **Keep it**, and rely on the published hardware floor.

Options 1 and 2 combine well. Option 3 is independent of both.

## B29: the efficiency term measures the tokenizer, not compute

**Today.**

```text
t̄ = (policy prefix tokens + content tokens + verdict tokens) / items
score = accuracy × (1 − λ·t)
```

`t` is `t̄` relative to the field's 90th percentile, and λ is 15%. The term switches off when `t̄` varies by less than 10% across the field, which under forced-choice decoding is the usual case. The tokens are counted by the miner's own tokenizer.

**Why it matters.**

- A coarse tokenizer produces fewer tokens for the same text, so it lowers `t` and can gain up to 15% without the model being any cheaper to run. The miner chooses the tokenizer.
- One model with an unusual tokenizer can push the spread over 10% and switch the term on for everyone.
- `[evaluation] batch_size` changes how the policy prefix is shared, and so changes `t̄` too.

**Options.**

1. **Estimate compute.** Parameters, from the checkpoint's config or weight bytes, times tokens under one fixed reference tokenizer. That's the same text and the same count for every model, so the estimate depends on model size.
2. **Count tokens with one reference tokenizer only.** Simpler, but it then measures nothing about the model, so the term would only reward shorter answers, and forced choice makes every answer one token.
3. **Drop the term** (`efficiency_lambda_bp = 0`) until a real compute measure exists. It's already off on most days because of the spread guard.
4. **Keep it**, accepting that it rewards tokenizer choice when it does switch on.

Option 3 is the smallest change. Option 1 is the measure the term was meant to be.
