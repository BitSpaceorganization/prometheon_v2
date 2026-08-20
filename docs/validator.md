# Validating

A validator runs **one cycle per day**, scoring the previous day at 04:00 UTC.

You pay for two things: ground-truth labelling (an OpenAI-compatible key) and
**the GPU that runs every miner's model**. Miners publish weights and pay for
nothing at evaluation time.

## Hardware

**Validating requires GPUs, and the floor is 8 × RTX 5090 (32 GB each, 256 GB
total) or better.** This is not a recommendation. Validators download every
eligible model and run it over the whole corpus, every day, and a machine that
cannot finish scores the field on whatever it completed — which pays miners for
the validator's hardware rather than for their models.

Where the numbers come from:

- **32 GB per card** is what a submitted model needs. Weights are capped at
  24 GiB, and the remainder is KV cache and activations. One model fits one
  card, which is what makes the cap meaningful.
- **Eight cards** is what a *field* needs. The corpus is thousands of items and
  every eligible model is run over all of it. One card is enough to evaluate one
  model, not to get through a day.

Fewer cards does not fail loudly. It runs, takes longer, and starts hitting
`model_timeout_seconds` — at which point models are scored on the fraction they
completed and the rest counts against them. A validator that is quietly too slow
still submits weights, and those weights are wrong.

**A known limit, stated plainly: the runtime evaluates models one at a time on
one device.** `device` selects where (`auto`, `cuda:0`, `cpu`); it does not yet
shard one model across cards or run several models in parallel. Eight cards is
therefore headroom for a growing field and for the throughput the day requires,
not something this version already exploits. Using it properly is future work,
and the requirement is set for where the subnet is going rather than where the
code is today.

**CPU is not a fallback for a real cycle.** `device = "auto"` resolves to CPU
when no CUDA device is present, which keeps a dry run possible on a laptop and
is unusably slow for anything else.

---

## Setup

```bash
git clone https://github.com/BitSpaceorganization/prometheon_v2
cd prometheon_v2
uv sync

cp configs/mainnet.example.toml ~/prometheon-mainnet.toml
$EDITOR ~/prometheon-mainnet.toml        # set [wallet] name and hotkey

export OPENAI_API_KEY="…"     # your own key — you pay for ground-truth labelling
```

**There is no inference credential any more.** Nothing is called over the
network to evaluate a model: the checkpoint is fetched from Hugging Face at the
committed SHA and run here. Miners share nothing with you, and there is no key
to be authorised for.

Your hotkey must be registered on the netuid and hold a validator permit. There
are no API keys for the subnet DB layer: every request is signed with your
hotkey and checked against the metagraph, so a deregistered validator loses
access the moment the chain says so.

### Run one cycle without touching anything

```toml
[validator]
dry_run = true
```

Everything runs: labelling, evaluation, scoring. Nothing is submitted or
published. This is the right first run.

```bash
uv run prometheon validator run --config ~/prometheon-mainnet.toml
```

---

## What a cycle does

```text
1. check the chain gates          ← before spending a penny
2. fetch the day's snapshot       ← from the DB layer, hash-verified
3. read commitments from chain    ← not from the DB layer
4. verify each model              ← Hugging Face: manifest, architecture, size
5. label the corpus               ← ground truth, fixed before any model sees it
6. evaluate every eligible model  ← each on its own view of the corpus
7. score and combine              ← one weight vector
8. submit weights, publish record
```

The order matters.

**Gates first.** Commit-reveal, the weights version key, and `mechid` support are
checked before a single API call is paid for. A cycle that could not have
submitted its result must not discover that after fourteen hours and a labelling
bill.

**The chain is the authority on commitments.** The DB layer says who is
*eligible*; the chain says what they *committed*. A DB layer that invented a
commitment could not get a model scored, because the commitment is read from
chain and then verified against Hugging Face independently. The two are
compared, and a disagreement is recorded in the published result. The chain
value is the one used.

**Ground truth is fixed before evaluation.** No model can influence what it is
measured against.

**Weights are resolved against a fresh metagraph.** A cycle takes hours, and a
UID is reassigned the moment its holder deregisters. Submitting against the
snapshot the cycle *scored* would pay whoever inherited the slot. Hotkeys that
vanished mid-cycle are dropped and their share goes to the burn hotkey rather
than being redistributed. Redistributing would let one deregistration quietly
raise everyone else's payout.

---

## What stops a cycle, and why

These fail loudly rather than producing a plausible-looking result. Each one
exists because the quiet version of it is worse:

| Condition | Why it is fatal |
|---|---|
| Commit-reveal enabled on the subnet | The SDK reroutes `set_weights` rather than refusing it, so weights land on a schedule this runtime does not model, and the submission still reads as a success |
| Weights version key mismatch | Weights under a stale key are accepted and then ignored |
| SDK does not accept `mechid` | Weights land on whatever mechanism the chain defaults to, and nothing reports it |
| Metagraph came back empty | A live subnet always has its owner registered, so this is a failed read. Treating it as real would mark every miner deregistered and burn the day |
| Labelling failed 3 batches in a row | An endpoint fault, not a corpus fault. Continuing produces an empty ground truth that scores every miner against nothing |
| Labelling excluded >20% of the corpus | What is left is not a benchmark |
| A low-base-rate batch came back unanimously violating | Evidence the labeller stopped labelling; publishing it would poison the day |
| Snapshot version or content hash mismatch | You were served something other than what the manifest describes |

What does **not** stop a cycle: any single miner. A model that will not
download, will not load, runs out of memory, or exceeds its time budget is
scored incorrect for those items and the cycle continues. That is a promise the code keeps. Every validator evaluates the same
miners at the same time, so an exception escaping one miner's evaluation would
take down the whole subnet's submission simultaneously.

---

## Reading the summary

```text
day                2026-08-05
snapshot           4f2a…  ← the DB layer's manifest hash
corpus             9c81…  ← hash over what you actually labelled
corpus items       9840
labelled           9840 ok, 12 excluded
labelling calls    104
models eligible    7/11
models evaluated   7
efficiency lambda  0bp    ← the spread guard fired; efficiency was off
token spread       4.10%
weighted hotkeys   9
burn               0 units to 5Owner…
```

Two hashes, answering two different questions. **snapshot** says *we were given
the same data*; **corpus** says *we made the same labelling calls out of it*. If
two validators' snapshot hashes match but their corpus hashes differ, they
disagreed about labels. That is expected on borderline items and is what
consensus arbitrates. If the snapshot hashes differ, the DB layer served two
different corpora. Report that one.

`efficiency lambda 0bp` is normal. Under forced-choice decoding every model
emits one completion token, so the only variation is tokenizer efficiency, a
narrow spread. When the coefficient of variation falls below 10%, the term is
switched off rather than allowed to randomise close rankings.

---

## Upgrading an existing validator

**`git pull` does not touch your config.** You copied
`configs/mainnet.example.toml` once; that copy keeps whatever values it had, so
a pull brings new code and new docs but leaves your `[scoring]` and
`[labelling]` behind. After pulling, diff your config against the example:

```bash
git pull
diff <(grep -vE '^\s*(#|$)' configs/mainnet.example.toml) \
     <(grep -vE '^\s*(#|$)' ~/prometheon-mainnet.toml)
```

Two of those differences change what you submit:

| Setting | If yours is stale | Consequence |
|---|---|---|
| `miner_burn_share_bp`, `dataset_share_bp` | `6000` / `5000` | you compute a **different weight vector** from every updated validator, and disagree on chain |
| `temperature` under `[labelling]` | absent | the cycle dies at labelling with HTTP 400 |

The runtime now refuses a model/temperature pair the endpoint would reject, so
the second one fails immediately at startup rather than hours into a cycle. The
first cannot be caught that way — a stale split is valid config, just a
different policy — so it is on you to diff.

You also need the hourly re-post entry below; a pull does not add it to your
crontab.

---

## Running it daily — **and re-posting between cycles**

Two entries, not one. The cycle runs once a day; the re-post runs every hour.

```cron
# The cycle: label, evaluate, score, submit. Once a day.
0 4 * * * cd /opt/prometheon_v2 && /usr/local/bin/uv run prometheon validator run \
    --config /etc/prometheon/mainnet.toml >> /var/log/prometheon.log 2>&1

# The re-post: send the same vector again so it keeps counting. Every hour.
30 * * * * cd /opt/prometheon_v2 && /usr/local/bin/uv run prometheon validator resubmit \
    --config /etc/prometheon/mainnet.toml >> /var/log/prometheon.log 2>&1
```

**Without the second entry your miners earn nothing for most of the day.**
Weights stop counting toward consensus once `activity_cutoff` passes — 720
blocks on netuid 108, about 2.4 hours — while a cycle runs every 24. For the
remaining ~21 hours the validator's row is masked out of consensus, and the
miners it weighted sit at zero incentive and zero emission however carefully
the cycle scored them.

`validator resubmit` never labels and never evaluates, so it costs nothing
beyond the extrinsic: it re-sends the allocation the last cycle computed. It
does re-read the chain, so the submission gates and the metagraph are resolved
fresh and a miner that deregistered since is dropped exactly as it would be at
first submission. Re-posting inside the 100-block (~20 min) weights rate limit
— which happens when the hourly entry lands just after the daily one — is
reported and skipped, not failed.

Hourly is deliberate: comfortably inside the 2.4-hour cutoff, and far enough
above the 20-minute rate limit that the two never fight.

Results go to stdout, progress to stderr, so a log keeps both and a pipe keeps
only the result.

Keep `submit_results = true`. Publishing your signed record is what lets anyone
else recompute your cycle from the same snapshot and check your arithmetic.
Turning it off makes your validator unauditable.

---

## Costs

Labelling is the bill you control. It scales with corpus size, and the policy
prefix is sent with every batch, roughly 2,100 tokens against a 4,096 ceiling.
Batching at 100 items amortises it; the default is already 100.

Evaluation costs GPU time rather than API calls, and it is now **your** GPU
time: each eligible model is downloaded once, cached by revision, loaded, run
over the corpus, then freed before the next.

That is the trade this design makes, and it is worth being explicit about who
pays what. Miners pay nothing to be evaluated — no serving, no endpoint, no GPU
bill between cycles. Validating carries the whole inference cost instead, which
is why the hardware floor above is a requirement rather than advice. Bandwidth
is the smaller half: models are capped at 24 GiB and cached by revision, so a
field that does not re-commit costs one download each.
