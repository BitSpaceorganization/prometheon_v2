# Mining

**40% of miner emission burns to the owner (uid 0) every day.** Miners compete
for the remaining **60%**, from two independent streams:

| Stream | Share of the 60% | Of total | What it pays for |
|---|---|---|---|
| Dataset contribution | two thirds | **40%** | test content your Fan Group users write |
| Model performance | one third | **20%** | how well your model judges everyone else's |

Data is weighted twice as heavily as model performance. It is what every model
is measured against, it accrues daily from real users, and it is the half you
can earn with no deployed model at all.

**One entry gate, two streams behind it.** A Fan Group makes you a miner when it
clears **both** floors: **at least 50 registered members** — accounts that have
joined the group — **and at least 25 active members**, members whose 7-day
activity score clears the platform's active threshold. The registered headcount
alone is not enough; the active floor is the anti-farming half, and it is what a
sign-up farm fails. It is not a condition on one stream or the other — below the
gate you are not on the eligible list, and a validator never looks at you.

Above it, the two streams are independent. Contribute data and deploy no model
and you earn the dataset half; deploy a model and your users submit nothing that
day and you earn the model half. Doing both is how you earn the most.

---

## 1. Register your Fan Group on bitfan.ai

Everything else depends on this, and it happens on the platform rather than
through any subnet command.

1. Sign in at **[bitfan.ai](https://bitfan.ai)**.
2. Create a Fan Group and grow it to **at least 50 registered members**
   (accounts that have joined your group), **at least 25 of them active** —
   members whose 7-day activity score clears the platform's active threshold.
3. Connect your **Talisman** wallet (or another Substrate wallet — polkadot-js,
   SubWallet, Nova) with the **Connect Wallet** button in the bitfan.ai site
   header, and sign the ownership proof it prompts for. The proved address is the
   hotkey you will mine with, and it becomes your Fan Group's leader hotkey on
   chain.

The signed proof is what makes your users' test content attributable to your
hotkey. Until it exists the platform has no way to credit you, so your
submissions count toward nobody and you earn nothing from either stream.

Your users then write posts and replies they believe violate
[`content_policy.md`](../content_policy.md), marked as **test content**. Those
are recorded and never published. They become the evaluation corpus every miner's
model is measured against, which is what the dataset half of the reward pays for.

---

## 2. Build a model

Any open-source causal language model **whose weights fit 24 GiB**. It is
judged on one task: given the policy and one piece of content, answer `YES` or
`NO`.

That ceiling is not a style guide, it is the deal. **Validators download your
model and run it on their own hardware**, so its size decides what validating
this subnet costs. 24 GiB leaves headroom on a 32 GB card, which is the floor a
validator is expected to own — an 8B checkpoint in fp16 sits comfortably inside
it. A model over the line is refused at commit rather than quietly costing
every validator an hour of bandwidth they cannot use.

**The evaluation runtime pins `transformers>=4.44,<4.47`, so your architecture
has to be one that release recognises.** `Qwen2.5` (`model_type: qwen2`) loads;
`Qwen3` does not, however new and capable it is — it arrived in transformers
4.51.

`prometheon model commit` and `model verify` both refuse an architecture the
runtime cannot load, before anything is written on chain:

```text
error [registry.architecture_unsupported] …declares model_type 'qwen3', which
the evaluation runtime's transformers does not recognise…
```

It checks against a list frozen from the runtime's own `transformers`, not
whatever you have installed — a newer release locally would accept a newer
architecture that every validator would then fail to load.

You are not writing inference code. Every model on the subnet is run by the
**same engine, on the validator's machine**, so a score difference is a model
difference and nothing else. What you control is the weights.

The engine decides by comparing the probability mass the model puts on `YES`
against `NO` at a single position. There is no free generation, so:

- a refusal is impossible, because there is no text to refuse with
- verbosity costs you nothing and buys you nothing
- a model that is confidently wrong loses to one that is quietly right

Train accordingly. What matters is calibration on the boundary cases in
[`content_policy.md`](../content_policy.md), not instruction-following polish.

**Your Hugging Face repository holds weights, and that is all it needs to
hold.** No inference code, no wrapper, no deploy script. Validators run one
engine — theirs — over every model on the subnet, in one process on one
machine, which is what makes a score difference a weights difference.

You never deploy anything. There is no endpoint to keep warm, no GPU bill for
serving, and nothing to keep running between cycles: publish the weights,
commit the revision, and the model is evaluated wherever validators are.

---

## 3. Publish the weights

Push the model to a public Hugging Face repository containing **weights, config
and tokenizer, and nothing else**. Any other file and the repository fails the
manifest, which a validator checks before it downloads anything:

```text
config.json  generation_config.json  tokenizer.json  tokenizer_config.json
special_tokens_map.json  added_tokens.json  vocab.json  merges.txt
tokenizer.model  chat_template.jinja  model.safetensors
model-00001-of-0000N.safetensors  model.safetensors.index.json
README.md  LICENSE  .gitattributes  .gitignore
```

Weights must be `safetensors`. Benchmark scripts, training args, adapter files
and original-format checkpoints all fail the manifest — most published
moderation models carry at least one of them, so check before you commit.

Two things worth confirming while you still have a choice, both of which a
validator will check and neither of which produces a useful error later:

```bash
# does the evaluation runtime know this architecture at all?
python -c "import json,urllib.request as u; \
  print(json.load(u.urlopen('https://huggingface.co/api/models/<you>/<model>'))['config']['model_type'])"

# do the weights fit the ceiling?
python -c "import json,urllib.request as u; \
  d=json.load(u.urlopen('https://huggingface.co/api/models/<you>/<model>?blobs=true')); \
  print(sum(s['size'] for s in d['siblings'] if s['rfilename'].endswith('.safetensors'))/1024**3, 'GiB')"
```

`prometheon model verify` runs the validator's own checks against what you
committed, and answers both properly.

---

## 4. Commit on chain

This is the submission. Until it lands, you are not scored on a model.

```bash
uv run prometheon model commit --config ~/prometheon-mainnet.toml \
    --hf-repo <you>/<model> --hf-revision <sha>
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
4. the architecture is one the evaluation runtime can load
5. the weights fit the 24 GiB ceiling
6. no earlier miner committed the same revision

Checks 4 and 5 exist because every validator runs your model. Neither is about
your judgement: a model that will not load, or will not fit, is work the whole
subnet attempts and fails at on the same day. Both are answered from the
Hugging Face API before anything is downloaded.

A check that used to be here is gone, and its absence is the point. While
miners served their own models, a commitment could pin an immutable SHA while
the deployment served a branch you could move afterwards, so the deployed
source had to be read and hashed to catch it. Validators resolve the SHA
themselves now: what is scored is what you committed, because the validator is
the one that fetched it.

Check 6 groups on the **revision SHA alone**, not on the repo name. Mirroring
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

**Answering `YES` to everything.** Production content supplies negatives, and so
does every test submission the labeller judged non-violating, so a degenerate
model is wrong on all of them. (Accuracy is prevalence-sensitive
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
