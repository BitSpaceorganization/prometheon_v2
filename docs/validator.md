# Validating

A validator scores **yesterday, once a day**, at 04:00 UTC by default, and **re-posts the same weights every 30 minutes** until the next cycle.

You pay for two things: ground-truth labelling (an OpenAI-compatible key) and **the GPUs that run every eligible model**. Miners publish weights and pay nothing at evaluation time.

The owner team runs a validator too, with this same CLI.

> **Before the subnet is registered** there's nothing to validate on a public network. To see a whole cycle work, with real extrinsics, run it on a local chain: [`localnet.md`](./localnet.md).

## Hardware

**This applies to `score_source = "local"`, the default, where the validator measures the field itself.** `score_source = "endpoint"` submits a record another validator published and needs no GPU at all. See [Two ways to run a validator](#two-ways-to-run-a-validator), and read what mirroring costs consensus before choosing it.

**Validating locally needs GPUs. The floor is 8 × RTX 5090 (32 GB each, 256 GB in total) or better.** Every eligible paid model is downloaded and run over the whole corpus every day. A machine that can't finish scores the field on whatever it completed, which pays miners for your hardware, not for their models.

- **32 GB per card** is what one model needs. Weights are capped at 24 GiB, and the rest is KV cache and activations.
- **Eight cards** is what a field needs. One card evaluates one model; it doesn't get through a day.

**A known limit:** the runtime evaluates models one at a time on one device. `device` picks it (`auto`, `cuda:0`, `cpu`, `mps`). It doesn't shard a model across cards or run several at once yet, so the eight cards are headroom for a growing field, not something this version already exploits.

`device = "auto"` falls back to CPU when there's no CUDA device. That's enough for a dry run on a laptop, or a two-model local chain, and far too slow for a real day.

---

## Setup

```bash
git clone https://github.com/BitSpaceorganization/prometheon_v2
cd prometheon_v2
uv sync --extra wrapper          # torch and transformers: needed to run models

cp configs/finney.example.toml ~/prometheon.toml
$EDITOR ~/prometheon.toml
export OPENAI_API_KEY="..."      # your own key; you pay for labelling
```

**`--extra wrapper` isn't optional in local mode.** Without it there's nothing to run models with. The cycle checks before it labels anything (whenever there's a model to run) and stops with `validator.fault`. Earlier versions scored every model zero instead and submitted that.

Fill in the config:

| Setting | What |
|---|---|
| `[chain] netuid` | **Required, no default.** The team announces it when the subnet is registered |
| `[chain] network` | `finney`, or your own node's `ws://` / `wss://` endpoint |
| `[wallet]` | your validator's wallet name and hotkey |
| `[hub] base_url` and `[hub.epoch_keys]` | the hub's address, and its Ed25519 public key under its key id. Both are published with the launch. Every epoch's signature is checked against this key |
| `[labelling]` | your labelling endpoint and model. The comments in the example explain the temperature rule |
| `[scoring]` | **leave as shipped.** These decide payouts, and every validator must run the same values |

Your hotkey must be registered and hold a validator permit. There are no API keys for the hub. Each request is signed with your hotkey and checked against the chain, so access follows registration.

### First run: change nothing on chain

```toml
[validator]
dry_run = true
```

Everything runs, including labelling, evaluation and scoring. Nothing is submitted or published.

```bash
uv run prometheon validator run --config ~/prometheon.toml
```

---

## What a cycle does

```text
1. check the chain gates            commit-reveal, weights version, mechid, owner hotkey
2. fetch the signed epoch           verify the hub's signature against your pinned key
3. download the day's two files     check sha256 and line count; reject malformed lines
4. decide who is eligible           ≥10 verified active members and ≥30 members
5. verify the paid models           Hugging Face: manifest, architecture, size, fingerprint
6. check you can run a model        a tiny canary on your device and dtype
7. remove near-duplicates           across the whole day's test content
8. label the corpus                 ground truth, fixed before any model sees it
9. run every eligible model         each on its own view of the corpus
10. score, split and save           the vector is written to disk before anything is sent
11. wait for the rate limit, send   then sign the record once and post it to the hub
```

- **Gates first.** Anything that would make the result unsubmittable is checked before anything is paid for.
- **The epoch is the source.** You score the commitments the hub froze, as the signed epoch lists them, and never the live chain values. Every validator scores the same models.
- **The hub is trusted for what it signs, and nothing else.** The epoch's member counts and payment status come from the hub. Labels, model verdicts and weights are yours.
- **Weights resolve against a fresh metagraph.** A cycle takes hours, and a uid can change hands. A hotkey that left before the send loses its units to the burn. They're never shared out among the others.

## What stops a cycle

These fail loudly rather than produce a plausible-looking result:

| Condition | Why |
|---|---|
| Commit-reveal is on, the weights version key mismatches, or the SDK can't name `mechid` | the weights would land somewhere the runtime doesn't model, and still read as a success |
| The metagraph came back empty | a live subnet always has its owner registered, so this is a failed read |
| The epoch's signature doesn't verify, or it's for another day, netuid or scoring version | you were served something other than the day you asked for |
| A miner row has no member count, or more verified members than members | the counts decide eligibility. Reading a missing one as zero would burn the day |
| A file's sha256 or line count differs from the epoch | the content isn't what the hub signed |
| Every test line in a non-empty file is rejected | a feed format problem, not a day with no data. Burning would hide it |
| Your `content_policy.md` hashes differently from the epoch's | you'd be labelling against different rules from everyone else |
| Hugging Face can't be reached to verify a model | marking the model invalid on this validator alone would split its vector from everyone else's |
| The canary can't run a model on your device | `validator.fault`. Fix your install or drivers; nothing is submitted |
| Labelling failed 3 batches in a row, each holding more than one submitter's content | an endpoint fault, not content |
| More than 20% of the production content excluded, or most of the test content lost for at least 3 submitters and at least half of them | what's left isn't a benchmark, and no single submitter could cause it |
| Production posts still come back violating when each is asked about alone: 10 or more in one batch, or more than 90% of at least 20 over the day | the labeller has stopped labelling |
| The vector can't be bent to the subnet's `max_weight_limit` without paying uids nobody scored | the owner has to raise the limit; see [`scoring.md`](./scoring.md#the-subnets-weight-limits) |

**What doesn't stop a cycle: any single submitter's content.** Items the provider refuses, or that never label cleanly, are excluded and cost their submitter, who keeps them in `S` with neither `V`. A batch that comes back unanimously violating against a low base rate is asked again one item at a time, where no item can reach another's verdict, and the summary counts those batches.

**Nor does any single miner's model.** A model that won't download, won't load, runs out of memory or times out is scored on what it completed, and one that never produced a verdict is published as `unreachable`. A repository that was deleted or made private is `invalid` for the day. The rest of the field is scored as normal.

## Reading the summary

```text
day                2026-11-05
epoch              8ca6ef49…   ← sha256 of the signed epoch bytes
frozen block       123456
corpus             9c81…       ← hash over what you labelled
near-duplicates    14 removed
eligible miners    9/31
corpus items       10240
labelling calls    103
models verified    6/7
models evaluated   6
efficiency lambda  0bp         ← the spread guard fired; usual under forced choice
token spread       4.10%
weighted hotkeys   10
burn               0 units to 5Owner…
published rows     11
```

The two hashes answer two questions:

- The **epoch** hash says you were given the same day as everyone else.
- The **corpus** hash says you labelled the same items in the same order.

If two validators' epoch hashes differ, one of them was served a different day. Report that.

---

## The state file, and running a day again

The computed vector is saved to `<state_directory>/last-weights.json` **before** it's sent. The file is written atomically. Then:

- If the send is refused, for example inside the rate limit, the day isn't lost. The next `validator run` for that day sends the saved vector instead of computing it again.
- A day that was computed **and** sent is finished. Running it again does nothing. Use `--force` to compute it again from scratch.
- The record is signed once and kept. If posting it fails, `validator resubmit` re-posts the same bytes. The hub treats an identical resend as a duplicate and refuses a different record for the same day, so the first record stands.
- The record always describes the vector actually sent, including units that went to the burn because a hotkey left.

## Running it daily, and re-posting every 30 minutes

Two jobs, not one:

```cron
# The cycle: label, evaluate, score, submit. Once a day.
0 4 * * * cd /opt/prometheon_v2 && /usr/local/bin/uv run --extra wrapper prometheon validator run \
    --config /etc/prometheon/prometheon.toml >> /var/log/prometheon.log 2>&1

# The re-post: send the same vector again so it keeps counting. Every 30 minutes.
*/30 * * * * cd /opt/prometheon_v2 && /usr/local/bin/uv run prometheon validator resubmit \
    --config /etc/prometheon/prometheon.toml >> /var/log/prometheon.log 2>&1
```

Or use systemd: [`deploy/systemd/`](../deploy/systemd/README.md). Or, where there's neither cron nor systemd, let the CLI keep the clock:

```bash
uv run --extra wrapper prometheon validator schedule --config /etc/prometheon/prometheon.toml
```

One process runs both jobs: the cycle at 04:00 UTC (`--cycle-hour` moves it) and the re-post at :00 and :30.

- **It runs yesterday's cycle whenever that day isn't finished.** After a restart past 04:00 it picks up an interrupted or missed cycle straight away instead of waiting a day. Older days aren't backfilled.
- **A failed cycle is retried, at most 3 times.** An epoch the hub hasn't built yet doesn't count as a failure.
- **Re-posts keep running while a cycle is busy.** A day-long cycle doesn't leave your weights to expire.

Run it under a supervisor: a loop that dies stops re-posting silently.

**Why re-post at all.** Your weights stop counting toward consensus once the subnet's `activity_cutoff` passes. The cycle runs once every 24 hours. Without re-posts, the miners you weighted earn nothing for most of the day. A re-post never labels and never evaluates. It re-reads the chain, so a hotkey that left since is dropped exactly as it would be at first submission. A re-post inside the weights rate limit is skipped, not failed.

**Pick the interval from the hyperparameters** (`btcli subnet hyperparameters --netuid <netuid>`). It must be shorter than `activity_cutoff`, with room for a missed post, and longer than `weights_set_rate_limit`. Thirty minutes suits a rate limit of 100 blocks (about 20 minutes) and a cutoff of a few hours. Check both once the subnet is registered.

`submit_weights = false` stops re-posts too. `validator resubmit` also says so when the saved vector is more than a day old.

Keep `submit_results = true`. Your signed record is what lets anyone recompute your cycle from the same epoch.

---

## Two ways to run a validator

The choice is `score_source` under `[scoring]`.

| | `local` (default) | `endpoint` |
|---|---|---|
| Labelling bill | yours | none |
| GPU | 8 × RTX 5090 or better | none |
| Cycle time | hours | seconds |
| What you submit | what you measured | what another validator measured |
| Contribution to consensus | an independent opinion | a copy of one |

### What mirroring costs the subnet

Consensus works by validators disagreeing when one of them is wrong. A field where most weight mirrors one provider **can't detect that provider being wrong**. The agreement is manufactured, and vtrust reads a captured subnet as a healthy one. The provider's signature proves which hotkey computed a vector. It never proves the vector is right.

### Running in endpoint mode

```toml
[scoring]
score_source = "endpoint"
score_provider = "5..."          # the validator hotkey whose record you mirror
```

```bash
uv sync                          # no --extra wrapper: nothing is run locally
uv run prometheon validator run --config ~/prometheon.toml
```

`validator run` notices the mode. It fetches `GET /api/v1/results/{day}/{score_provider}` from the hub, and refuses to submit unless all of these hold:

| Check | Why |
|---|---|
| Published by the pinned `score_provider` | otherwise the hub chooses whom you trust |
| Signature verifies against that hotkey | the hub can't mint a record for a key it doesn't hold |
| Same day, netuid and scoring version | catches a stale, misrouted or old-rules record |
| Provider is registered and holds a permit | a deregistered validator's scores aren't worth submitting |
| The record's epoch hash matches the signed epoch for the day | catches a record computed from a different day |

The burn in the record is submitted as part of the vector. Dropping it would scale every miner up. `validator resubmit` in this mode also checks for a newer day from the provider and catches up, so a provider that published late doesn't leave you a day behind.

You still need a registered hotkey with a validator permit, and you still need the re-post. Choosing a provider is a trust decision, so no default is filled in. The hub's public pages list who has published.

---

## Costs

Labelling is the bill you control. It scales with the corpus, and the policy goes with every batch, so batches of 100 items (the maximum and the default) keep the overhead down.

Evaluation is your GPU time. Each eligible paid model is downloaded once per revision, loaded, run over the corpus and freed before the next. Only paid models of eligible miners are run, and the submission fee keeps the field to models someone paid to enter.

Disk is the other limit: the download cache keeps every revision it has seen. With `[evaluation] discard_checkpoints = true`, each model is deleted after scoring, so the disk needs to hold only the largest model rather than the whole field. Files another revision still uses, such as an unchanged tokenizer, are kept.

## Upgrading

`git pull` doesn't touch your config. If a release changes the rules, it changes `scoring_version`, and a config written for the old rules refuses to load until you update it. After pulling, compare your config with the example:

```bash
git pull
diff <(grep -vE '^\s*(#|$)' configs/finney.example.toml) \
     <(grep -vE '^\s*(#|$)' ~/prometheon.toml)
```
