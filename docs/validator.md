# Validating

A validator scores **yesterday, once a day**, at 04:00 UTC by default, and **re-posts the same weights every 30 minutes** until the next cycle.

You pay for two things: ground-truth labelling (an OpenAI-compatible key) and **the GPU that runs every eligible model**. Miners publish weights and pay nothing at evaluation time.

The owner team runs a validator too, with this same CLI.

> **Before the subnet is registered** there's nothing to validate on a public network. To see a whole cycle work, with real extrinsics, run it on a local chain: [`localnet.md`](./localnet.md).

## Hardware

**This applies to score mode, the default, where the validator does the day's work itself.** [Copy mode](#copy-mode) submits the owner validator's weights unchanged and needs no GPU and no OpenAI key. Read what it means before choosing it.

**Scoring the day yourself needs one GPU with 24 GB, for example an RTX 4090, and an OpenAI key for labelling.** Every eligible paid model is downloaded and run over the day's corpus, one model at a time.

- Models are capped at 4 billion parameters and 16 GiB of weights, and the validator loads them in 16-bit, so a model at the cap takes about 8 GB of the card. The rest is cache and activations.
- A model at the cap judges a day's corpus in minutes.
- Each model has 30 minutes (`[evaluation] model_timeout_seconds`) to load and judge its corpus. One that doesn't finish gets no score from you for the day (`timed_out`), and its unfinished items count for nothing, so a slow card can't count an honest model's items as wrong. The floor and the margin keep that from happening.

`device` picks the card (`auto`, `cuda:0`, `cpu`, `mps`). `device = "auto"` falls back to CPU when there's no CUDA device. That's enough for a dry run on a laptop, or a two-model local chain, and far too slow for a real day.

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

**`--extra wrapper` isn't optional in score mode.** Without it there's nothing to run models with. The cycle checks before it labels anything (whenever there's a model to run) and stops with `validator.fault`. Earlier versions scored every model zero instead and submitted that.

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
5. verify the paid models           Hugging Face: manifest, architecture, size, fingerprint, parameters
6. check you can run a model        a tiny canary on your device and dtype
7. label every post                 one request per post, never showing the author's claim
8. run every eligible model         each on its own view of the corpus
9. score, split and save            the vector is written to disk before anything is sent
10. wait for the rate limit, send   then sign the record once and post it to the hub
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
| The labelling endpoint refuses the request itself (a bad key, an unknown model, a malformed request), or keeps failing after its retries | an endpoint fault, not content |
| More than 20% of the production posts excluded, or most of the test posts lost for at least 3 submitters and at least half of them | what's left isn't a benchmark, and no single submitter could cause it |
| More than 90% of at least 20 labelled production posts come back violating | the labeller has stopped labelling |
| The vector can't be bent to the subnet's `max_weight_limit` without paying uids nobody scored | the owner has to raise the limit; see [`scoring.md`](./scoring.md#the-subnets-weight-limits) |

**What doesn't stop a cycle: any single submitter's content.** A post the provider refuses, or that doesn't label cleanly on a second ask, is excluded and costs its submitter, who keeps it in `S` with neither `V`. Each post is labelled in a request of its own, so no post can reach another's verdict. Rate limits and server errors are retried with backoff before a post is judged.

**Nor does any single miner's model.** A model that won't download, won't load or runs out of memory is scored on what it completed, and one that never produced a verdict is published as `unreachable`. One that runs out of time is published as `timed_out`, with no score. A repository that was deleted or made private is `invalid` for the day, and so is a model over 4 billion parameters. The rest of the field is scored as normal.

## Reading the summary

```text
day                2026-11-05
epoch              8ca6ef49…   ← sha256 of the signed epoch bytes
frozen block       123456
corpus             9c81…       ← hash over what you labelled
eligible miners    9/31
corpus items       1240
labelling calls    1243
models verified    6/7
models evaluated   6
models timed out   0
models ranked      5
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

One process runs both jobs: the cycle at 04:00 UTC (`--cycle-hour` moves it) and the re-post at :00 and :30. In copy mode, drop `--extra wrapper` from these commands: nothing is run locally.

- **It runs yesterday's cycle whenever that day isn't finished.** After a restart past 04:00 it picks up an interrupted or missed cycle straight away instead of waiting a day. Older days aren't backfilled.
- **A failed cycle is retried, at most 3 times.** An epoch the hub hasn't built yet doesn't count as a failure.
- **Re-posts keep running while a cycle is busy.** A day-long cycle doesn't leave your weights to expire.

Run it under a supervisor: a loop that dies stops re-posting silently.

**Why re-post at all.** Your weights stop counting toward consensus once the subnet's `activity_cutoff` passes. The cycle runs once every 24 hours. Without re-posts, the miners you weighted earn nothing for most of the day. A re-post never labels and never evaluates. It re-reads the chain, so a hotkey that left since is dropped exactly as it would be at first submission. A re-post inside the weights rate limit is skipped, not failed.

**Pick the interval from the hyperparameters** (`btcli subnet hyperparameters --netuid <netuid>`). It must be shorter than `activity_cutoff`, with room for a missed post, and longer than `weights_set_rate_limit`. Thirty minutes suits a rate limit of 100 blocks (about 20 minutes) and a cutoff of a few hours. Check both once the subnet is registered.

`submit_weights = false` stops re-posts too. `validator resubmit` also says so when the saved vector is more than a day old.

Keep `submit_results = true`. Your signed record is what lets anyone recompute your cycle from the same epoch.

---

## Copy mode

A validator that doesn't want to run the work can copy the owner validator's weights instead.

```toml
[validator]
mode = "copy"
copy_from = "5..."       # the owner validator's hotkey, published with the launch
```

```bash
uv sync                  # no --extra wrapper: no model is run
uv run prometheon validator run --config ~/prometheon.toml
```

**What it does.** Each day it fetches the owner validator's signed record from the hub and submits its weights unchanged. It labels nothing and runs no model, so it needs no GPU and no OpenAI key.

**What that means.** A validator in copy mode has no opinion of its own. **The owner validator's results decide the weights of every validator that copies it.** If the owner validator scores a day wrongly, every copy submits the same mistake, and consensus can't catch it, because the copies agree with the owner validator by construction.

| | `score` (default) | `copy` |
|---|---|---|
| Labelling bill | yours | none |
| GPU | one with 24 GB, for example an RTX 4090 | none |
| Cycle time | an hour or more, with the field | seconds |
| What you submit | what you measured | the owner validator's weights, unchanged |

`validator run` notices the mode. It fetches the record with `POST /api/v1/validator/weights`, signed with your hotkey like the epoch request, and refuses to submit unless all of these hold:

| Check | Why |
|---|---|
| Signed by `copy_from`, the owner validator's hotkey, and no other | copy mode copies the owner validator only, so the hub can't hand you anybody else's record, or one it made |
| The signature verifies against that hotkey | the hub can't make a record for a key it doesn't hold |
| Same day, netuid and scoring version | catches a stale, misrouted or old-rules record |
| The owner validator is registered and holds a permit | a deregistered validator's weights aren't worth submitting |
| The record's epoch hash matches the signed epoch for the day | catches a record computed from a different day |

The burn in the record is submitted as part of the vector. Dropping it would scale every miner up. Until the owner validator has posted the day, the hub answers "not ready", and the scheduler tries again without counting it as a failure. `validator resubmit` in copy mode also looks for a newer day and catches up, so an owner validator that posts late doesn't leave you a day behind.

You still need a registered hotkey with a validator permit, `[hub.epoch_keys]` pinned, and the re-post.

---

## Costs

Labelling is the bill you control. Each post is one request, and every request carries the policy, so a day costs about one policy's worth of input tokens per post. Providers that cache a repeated prompt prefix bill most of that at their cached rate. `[labelling] max_in_flight` (8) sets how many requests are open at once.

Evaluation is your GPU time. Each eligible paid model is downloaded once per revision, loaded, run over the corpus and freed before the next. A model at the 4-billion parameter cap takes minutes. Only paid models of eligible miners are run, and the submission fee keeps the field to models someone paid to enter.

Disk is the other limit: the download cache keeps every revision it has seen. With `[evaluation] discard_checkpoints = true`, each model is deleted after scoring, so the disk needs to hold only the largest model rather than the whole field. Files another revision still uses, such as an unchanged tokenizer, are kept.

## Upgrading

`git pull` doesn't touch your config. If a release changes the rules, it changes `scoring_version`, and a config written for the old rules refuses to load until you update it. After pulling, compare your config with the example:

```bash
git pull
diff <(grep -vE '^\s*(#|$)' configs/finney.example.toml) \
     <(grep -vE '^\s*(#|$)' ~/prometheon.toml)
```
