# Mining

**60% of miner emission burns to the owner (uid 0) every day.** Miners compete
for the remaining **40%**, from two independent streams in equal measure:

| Stream | Share of the 40% | What it pays for |
|---|---|---|
| Dataset contribution | 50% (20% of total) | test content your Fan Group users write |
| Model performance | 50% (20% of total) | how well your model judges everyone else's |

**One entry gate, two streams behind it.** A Fan Group with **at least 50
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
2. Create a Fan Group and grow it to **at least 50 qualified users**. Qualified
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

Any open-source causal language model that fits the RTX PRO 6000 pool the
wrapper deploys to (96 GB VRAM per GPU, 1–8 GPUs). It is judged on one task:
given the policy and one piece of content, answer `YES` or `NO`.

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

**Your Hugging Face repository holds weights, and that is all it needs to
hold.** The engine is not a file you copy in: it *is* the canonical wrapper you
deploy on Chutes, which loads your weights from the repo at the revision you
commit. One artefact, hashed in one place.

```bash
uv run prometheon canonical          # prints the wrapper hash validators accept
```

A validator fetches the wrapper source from your Chutes deployment, normalises
it, and compares the hash. Everything except the values you set in §3 is fixed,
so two miners running the same wrapper hash identically however they filled
those in — which is what makes a score difference a model difference.

---

## 3. Deploy behind the canonical wrapper

Render the wrapper for your deployment:

```bash
uv run prometheon model render --config ~/prometheon-mainnet.toml \
    --chutes-user <you> \
    --hf-repo <you>/<model> \
    --hf-revision <40-char-sha> \
    --output wrapper.py
```

The command hashes what it produced and refuses to write a wrapper no validator
would accept, so a mistake costs a second rather than a day. It also prints the
**chute id** you commit in §4 — derived from your Chutes account and hotkey, so
it is known before the chute exists.

You supply three values; the render fills them in and everything else is fixed:

```python
PROMETHEON_HF_REPO     = "you/model"      # --hf-repo
PROMETHEON_HF_REVISION = "<40-char SHA>"  # --hf-revision
PROMETHEON_CHUTES_USER = "you"            # --chutes-user
```

Your hotkey, read from `[wallet]`, names the chute. `--gpu-count` (1–8) and
`--min-vram-gb` (16–140) size the instance. None of these change the wrapper
hash. Everything else — the engine, the framing, the image, and the deployment
target below — is fixed, and changing it makes you ineligible.

**The revision must be a 40-character commit SHA, never a branch or tag.** A
branch moves, and a commitment that can move after it is verified is not a
commitment; the CLI rejects one before it reaches the chain.

### The deployment is confidential, on the pro_6000 pool

The wrapper pins two things you do not choose:

- **`tee=True`** — the chute runs in a trusted execution enclave. Chutes accepts
  only TEE chutes, and the subnet requires it: your weights are served from
  hardware the host cannot read into.
- **`include=["pro_6000"]`** — the RTX PRO 6000 pool, which is where TEE capacity
  is available. The wrapper targets it so a deploy is scheduled rather than
  refused for capacity.

Deploy from your own Chutes account — you own the chute and pay for its GPU hours:

```bash
chutes deploy wrapper.py --accept-fee
```

### Reachability

The chute is named **`prometheon-<hotkey>`**, which places it in the subnet's
`prometheon` namespace. Validators call your model through the subnet's Chutes
inference key, which is scoped to that namespace — so deploying under the
canonical name is all that is required. You never share a credential and never
hand one over. The caller pays per invocation; you pay for the GPU your chute
occupies.

---

## 4. Commit on chain

This is the submission. Until it lands, you are not scored on a model.

```bash
uv run prometheon model commit --config ~/prometheon-mainnet.toml \
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
uv run prometheon model verify --config ~/prometheon-mainnet.toml
```

This runs the validator's own eligibility pipeline, the same code a validator
runs rather than a miner-side approximation of it, against the commitment
currently on chain. It exits non-zero and names the first failing check if you
would not be scored.

The checks run cheapest-first, so the reason you get is the first thing you can
fix:

1. a commitment exists and decodes
2. its revision is a 40-character commit SHA
3. the Hugging Face repo holds model files and no executable code
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
uv run prometheon dataset pull --config ~/prometheon-mainnet.toml --date 2026-08-05
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
