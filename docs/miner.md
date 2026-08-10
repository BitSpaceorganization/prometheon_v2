# Mining

You earn from two independent streams, in equal measure:

| Stream | Share | What it pays for |
|---|---|---|
| Dataset contribution | 50% | test content your Fan Group users write |
| Model performance | 50% | how well your model judges everyone else's |

**One entry gate, two streams behind it.** A Fan Group with **at least 500
active members** — real accounts that have cleared the platform's anti-farming
checks — is what makes you a miner at all. It is not a condition on one stream
or the other; below it you are not on the eligible list, and a validator never
looks at you.

Above it, the two streams are independent. Contribute data and deploy no model
and you earn the dataset half; deploy a model and your users submit nothing that
day and you earn the model half. Doing both is how you earn the most.

---

## 1. Register your Fan Group on bitfan.ai

Everything else depends on this, and it happens on the platform rather than
through any subnet command.

1. Sign in at **[bitfan.ai](https://bitfan.ai)**.
2. Create a Fan Group and grow it to **at least 500 qualified users**. Qualified
   means active real accounts that have cleared the platform's anti-farming
   checks, not raw signups.
3. Open **[bitfan.ai/me/prometheon](https://bitfan.ai/me/prometheon)** and link
   your Fan Group leader account to the hotkey you will mine with.

The link is what makes your users' test content attributable to your hotkey. Until
it exists the platform has no way to credit you, so your submissions count toward
nobody and you earn nothing from either stream.

Your users then write posts and replies they believe violate
[`content_policy.md`](../content_policy.md), marked as **test content**. Those
are recorded and never published. They become the evaluation corpus every miner's
model is measured against, which is what the dataset half of the reward pays for.

---

## 2. Build a model

Any open-source model that fits on the GPU you are willing to pay for. It is
judged on one task: given the policy and one piece of content, answer `YES` or
`NO`.

You are not writing inference code. Every model on the subnet runs the **same**
hash-pinned engine, so a score difference is a model difference and nothing
else. What you control is the weights.

The engine decides by comparing the probability mass the model puts on `YES`
against `NO` at a single position. There is no free generation, so:

- a refusal is impossible, because there is no text to refuse with
- verbosity costs you nothing and buys you nothing
- a model that is confidently wrong loses to one that is quietly right

Train accordingly. What matters is calibration on the boundary cases in
[`content_policy.md`](../content_policy.md), not instruction-following polish.

**Your Hugging Face repository holds weights and the canonical `miner.py`, and
nothing else.** Any extra file and your deployment is rejected. Get the exact
engine to commit:

```bash
uv run prometheon canonical          # prints the engine and wrapper hashes
```

The engine file ships inside the package at
`src/prometheon/canonical/assets/miner.py.template`. Copy it into your repo as
`miner.py`, byte for byte. A validator recomputes its SHA-256 and compares.

---

## 3. Deploy behind the canonical wrapper

Render the wrapper for your deployment:

```bash
uv run prometheon model render --config ~/prometheon-testnet.toml \
    --chutes-user <you> \
    --hf-repo <you>/<model> \
    --hf-revision <40-char-sha> \
    --chute-id <uuid> \
    --output wrapper.py
```

The command hashes what it produced and refuses to write a wrapper no validator
would accept, so a mistake costs you a second rather than a day.

Four values are yours to set. Everything else is fixed, and changing any of it
changes the hash and makes you ineligible:

```python
PROMETHEON_HF_REPO     = "you/model"
PROMETHEON_HF_REVISION = "<40-char commit SHA>"
PROMETHEON_CHUTES_USER = "you"
PROMETHEON_CHUTE_ID    = "<uuid>"
```

**The revision must be a 40-character commit SHA, never a branch or tag.** A
branch moves. A commitment that can move after it is verified is not a
commitment, and the CLI rejects one before it reaches the chain.

Deploy it to Chutes with your own account, on whatever GPU you choose. The image
and node selector are yours, in your `chute_config.yml`. You are the chute owner,
so you pay for the GPU hours.

### Authorising the subnet

Validators call your model without ever holding your credentials:

```bash
chutes share --chute-id <your-chute-id> --user-id <subnet-user-id>
```

The caller pays per invocation; you pay for the GPU your chute occupies. No key
is exchanged in either direction.

---

## 4. Commit on chain

This is the submission. Until it lands, you are not scored on a model.

```bash
uv run prometheon model commit --config ~/prometheon-testnet.toml \
    --hf-repo <you>/<model> --hf-revision <sha> --chute-id <uuid>
```

Add `--dry-run` to see the exact 128-byte payload without writing it.

Two things to know before you commit:

**Commitments cost quota.** The commitments pallet charges every write against a
per-hotkey byte budget that resets each epoch. A commitment that does not fit,
because the repo name is long, is rejected by the chain rather than by us. The
CLI prints the size against the 128-byte ceiling.

**Re-committing re-dates your claim.** Priority over a duplicated model goes to
whoever committed the revision first, measured by the block of the *current*
commitment, and the chain resets that on every write. If you re-commit an
unchanged model, you move to the back of your own queue. Re-commit when the
model or the deployment actually changed, not otherwise.

---

## 5. Check what a validator will see

```bash
uv run prometheon model verify --config ~/prometheon-testnet.toml
```

This runs the validator's own eligibility pipeline, the same code a validator
runs rather than a miner-side approximation of it, against the commitment
currently on chain. It exits non-zero and names the first failing check if you
would not be scored.

The checks run cheapest-first, so the reason you get is the first thing you can
fix:

1. a commitment exists and decodes
2. its revision is a 40-character commit SHA
3. the Hugging Face repo passes the manifest and carries the canonical engine
4. the deployed chute source hashes to an accepted wrapper
5. the repo and revision *declared in that source* equal what you committed
6. the chute is hot
7. no earlier miner committed the same revision

Check 5 is the one that catches the clever failure: committing an immutable SHA
while serving a branch you can move afterwards. The revision is read out of the
deployed source, which check 4 has already proven is the canonical wrapper.

Check 7 groups on the **revision SHA alone**, not on the repo name. Mirroring
someone's repository preserves its commit SHAs exactly, so a mirror and its
original are the same model as far as this check is concerned, and the later
commit loses.

---

## 6. Train against released data

```bash
uv run prometheon dataset pull --config ~/prometheon-testnet.toml --date 2026-08-05
```

A day's test content is released **two days later**, after the evaluation that
used it has finished. You cannot see the corpus you are being measured on today,
so memorisation is not a strategy. Yesterday's is yours to train on.

---

## What will not work

**Watermarking your own submissions.** You are never scored on test content your
own users wrote. Recognising your own items buys nothing, because they are not
in your corpus.

**Answering `YES` to everything.** Production content supplies the negatives, so
a degenerate model is wrong on all of them. (Accuracy is prevalence-sensitive
and this stops being true if test positives ever come to swamp production
negatives. Validators log that ratio every cycle and the metric moves to
balanced accuracy before it bites. See [`scoring.md`](./scoring.md).)

**Answering fast instead of well.** Efficiency is capped at 15% and applies
multiplicatively, so it can never overturn an accuracy gap wider than that. It
also switches off entirely when the field's token spread is below 10%, which it
usually is.

**Reporting no token usage.** A model that reports nothing is charged the *full*
efficiency discount rather than none of it, and takes no part in setting the
scale.

**Submitting junk test content in volume.** Dataset score is `V²/S`: valid items
times submission accuracy. Fifty good items beat a hundred half-junk ones.

**Copying a model.** Two hotkeys committing the same revision SHA are one
submission; the later one is dropped.
