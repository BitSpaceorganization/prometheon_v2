# Mining

There are two ways to earn, and validators score both every day from the same signed record:

| Pool | Share of emission | What it pays for |
|---|---|---|
| Data | **70%** | test posts written by identity-verified members of your BitFan group, confirmed by the validator's labeller |
| Models | **30%** (25%, 4%, 1%) | the three best moderation models, run by validators on their own hardware |

You can earn from either pool or both. **One gate covers both:** your hotkey must be eligible for the day.

**Eligible** means the signed epoch lists your hotkey with **at least 10 identity-verified active members and at least 30 members in total**. Every member is phone-verified, and the 30 includes the 10. Both floors are validator config (`min_verified_active_members`, `min_members`). The counts come from BitFan, through the hub. Below the gate, your members' test posts still enter the corpus, because they're verified human content, but you earn nothing from either pool, and your model isn't evaluated.

The rules are exact in [`scoring.md`](./scoring.md). This page covers what to do.

## A day, in UTC

```text
day D        your members write test posts; you commit or update a model
D+1 00:00    the hub freezes the day at the first block at or after midnight
D+1 01:00    the signed epoch for D is published: miners, counts, frozen models, payment
D+1 04:00    validators label D, run every eligible paid model, set weights
every 30 min validators re-post the same weights until the next day's cycle
D+2 00:00    D's test posts become readable on the public viewer
```

A model committed during day D is in D's epoch if its commitment is on chain at the freeze.

---

## 1. Lead a group on BitFan

[BitFan](https://bitfan.ai) is a separate consumer product and Prometheon's first data partner. Your group's leader wallet must be your mining hotkey, registered on the subnet and not banned. BitFan's own documentation covers creating a group and linking the wallet.

**Only identity-verified people count.** BitFan's daily export includes only human accounts that passed identity verification (Stripe Identity) and aren't banned or on hold. Everyone else's posts never reach the feed. Each person can write at most 5 violating and 5 compliant test posts per UTC day.

Your members write posts and say, for each one, whether it violates [`content_policy.md`](../content_policy.md). Those are **test posts**. They're recorded, never shown publicly as posts, and they become part of the corpus every model is measured against.

## 2. The data pool, 70%

Each validator labels every test post itself, against the policy, with its own labeller, one post per request. The labeller never sees your members' claims, and they're never trusted as ground truth. For the day:

```text
S      = your test posts in the day's feed
V_pos  = posts claimed violating that the labeller found violating
V_neg  = posts claimed compliant that the labeller found compliant

score  = 2 × min(V_pos, V_neg) × (V_pos + V_neg) / S
```

The top 10 eligible miners with a score above zero split the 70% in proportion to score. If fewer than 10 qualify, those present share the whole pool.

What this means in practice:

- **Balance pays.** 40 violating and 0 compliant scores 0. 20 and 20 scores 40.
- **Accuracy pays.** Posts the labeller disagrees with stay in `S`, so they dilute the score.
- **Copies don't get in.** BitFan refuses a post that's a near-copy of one already written, whoever wrote it. Reposting someone else's post, or your own with different punctuation, never reaches the feed.

[`scoring.md`](./scoring.md#data-pool-70) has worked examples.

## 3. The model pool, 30%

### Build a model

Any open causal language model with **at most 4,000,000,000 parameters, and weights that fit in 16 GiB**. Validators download your model and run it on one 24 GB GPU, so its size decides what validating costs.

- **The cap is hard.** A model with more parameters is invalid, as too large, and is never evaluated. The hub doesn't list it, so it's never charged.
- **Size costs score.** Past 500 million parameters, your accuracy is reduced by one point for every doubling: `floor(100 × log2(P / 500,000,000))` basis points.

| Parameters | Penalty |
|---|---|
| up to 500M | none |
| 1B | 1 point (100 bp) |
| 1.5B | 1.58 points (158 bp) |
| 2B | 2 points (200 bp) |
| 4B | 3 points (300 bp) |

`P` counts every tensor in every `.safetensors` file you publish, read from the files' headers. If you publish the same weights twice, as one file and as shards, they're counted twice.

**The evaluation runtime pins `transformers>=5.20.2,<5.21`, so your architecture has to be one that release recognises.** Qwen3 (`model_type: qwen3`), Gemma 3 (`gemma3_text`, the 270M and 1B checkpoints), Llama 3.2 (`llama`) and Qwen2.5 (`qwen2`) all load. An architecture newer than that release doesn't: Flint (`flint`), for example, arrived in 5.21. `model commit` and `model verify` both refuse an architecture the runtime can't load, checked against a list frozen from the runtime's own transformers.

The engine also needs ordinary attention layers. It computes the policy once and shares that work across every item, and a layer that keeps recurrent state, as linear-attention and state-space models do, can't be shared that way. So a `mamba`, `jamba`, `falcon_h1`, `qwen3_next` or `qwen3_5` model loads but fails every batch and scores zero. `model verify` doesn't check for this yet, so check your `model_type` before you commit.

You don't write inference code. Every model runs on the **same engine, on the validator's machine**, so a score difference is a weights difference:

- The framing is fixed plain text, never your chat template.
- Items are judged one at a time, never concatenated.
- The verdict is **forced choice**: the probability on `YES` against `NO` at one position. There's no generation, so a refusal is impossible and verbosity buys nothing.

Train for calibration on the policy's boundary cases.

### Publish the weights

Use a public Hugging Face repository holding **weights, config and tokenizer, and nothing else**:

```text
config.json  generation_config.json  tokenizer.json  tokenizer_config.json
special_tokens_map.json  added_tokens.json  vocab.json  merges.txt
tokenizer.model  chat_template.jinja  model.safetensors
model-00001-of-0000N.safetensors  model.safetensors.index.json
README.md  LICENSE  .gitattributes  .gitignore
```

- **Weights must be `safetensors` stored in Git LFS.** Your model is identified by its *weights fingerprint*: sha256 over the sorted LFS sha256s of every `.safetensors` file. A weight file outside LFS can't be fingerprinted, and the repository is refused.
- `config.json` is required.
- Everything other than weights must total 64 MiB or less.
- At most 256 weight files.

Benchmark scripts, training arguments, adapter files and original-format checkpoints all fail the manifest.

### Pay for a submission credit

A new model costs **50 alpha**, paid in advance as a credit:

```bash
uv run prometheon credits balance --config ~/prometheon.toml
uv run prometheon credits buy --config ~/prometheon.toml --dry-run   # shows the transfer
uv run prometheon credits buy --config ~/prometheon.toml             # sends it, after you type yes
```

`credits buy` moves 50 alpha of your stake on the subnet, from your coldkey to the treasury coldkey, with `transfer_stake` on the same netuid. It checks that the stake is there first, and sends nothing without your confirmation. The treasury address is `[credits] treasury_coldkey` or `PROMETHEON_TREASURY_COLDKEY`, published with the launch. Never send a fee anywhere else.

- One credit per 50 alpha. Remainders carry forward.
- Credits belong to the **coldkey** that paid, and every hotkey under it can use them.
- **A credit is spent only when the model is evaluated:** at the first freeze where your coldkey has a free credit, your miner is eligible, and the model passes the model checks. After that, the same `(hotkey, weights fingerprint)` costs nothing again. **Re-committing the same weights costs nothing.** New weights are a new fingerprint and cost a new credit.
- A model committed without a credit is listed as `unpaid` and isn't evaluated. It's picked up at the next freeze after a credit arrives.
- A covered model whose miner isn't eligible that day isn't evaluated, and **its credit isn't spent**. The credit waits for a day the model is evaluated.
- The hub lists only models that pass its own copy of the registry checks (manifest, at most 256 weight files, a known `model_type`, the 16 GiB and 64 MiB ceilings, weights in LFS, and at most 4,000,000,000 parameters), so a model that fails them is never charged.
- **There are no refunds.** The ledger is public: `prometheon credits balance` shows each entry with its block number and extrinsic hash.

### Commit on chain

```bash
uv run prometheon model commit --config ~/prometheon.toml \
    --hf-repo <you>/<model> --hf-revision <40-character sha>
```

`--dry-run` prints the payload without writing it. The commitment is limited to 128 bytes, and the CLI prints the size. `--if-changed` writes only when the chain holds something different. Use it for anything on a timer: an unchanged re-commit changes nothing but still pays a transaction fee.

Validators never read your live commitment. They score the hub's frozen copy, dated by the **first** block at which your hotkey committed those weights.

### Check what a validator will conclude

```bash
uv run prometheon model verify --config ~/prometheon.toml
```

This runs the validator's own checks against your commitment on chain, or against `--hf-repo`/`--hf-revision` if you pass them. It prints your weights fingerprint. It exits non-zero, naming the first failing check, if your model wouldn't be evaluated:

1. a commitment exists and decodes
2. its revision is a 40-character commit SHA
3. the repository exists, is public, and its files pass the manifest, including `config.json`
4. the architecture is one the runtime can load
5. the weights fit 16 GiB, and everything else fits 64 MiB
6. every weight file is in Git LFS, so the weights can be fingerprinted
7. the model has at most 4,000,000,000 parameters

It prints the parameter count `P` and the size penalty your model will carry, so you see both before you pay for a credit.

On a validator, two more checks run against the epoch. Your fingerprint must match the one the hub froze. And among models with the same fingerprint, only the earliest `(first commit block, hotkey)` is evaluated. The others are `duplicate`.

### How models are ranked

Each eligible paid model runs over the day's labelled corpus: everyone's test posts plus a sample of about 200 public BitFan posts. **Test posts from your own coldkey are excluded**, from every hotkey under it, so recognising your own posts buys nothing.

```text
raw accuracy = correct / scored, in basis points
score        = raw accuracy − size penalty, never below 0
```

- A model scored on fewer than 100 items isn't ranked.
- A score of zero is never paid.
- **The earlier model keeps its place unless it's beaten by more than a point.** For first place, the validator takes every model scoring within 100 bp of the best score, and the one committed first wins: the earliest first-commit block, then the lower hotkey. It then does the same for second and third among the models left. A copy or a small fine-tune of a ranked model has to beat it by a full point to take its place.
- The first three places pay 25%, 4% and 1% of emission. A place nobody fills goes to the lowest-placed winner present: two models get 25% and 5%, and one model gets 30%.
- **One time limit.** Your model has 30 minutes on each validator to load and judge its whole corpus. If it hasn't finished, it gets no score from that validator for the day, shown as `timed_out`. A model within the size cap needs minutes on a 24 GB card.

## 4. Released test content

A day's test posts become readable two days later on the hub's public viewer, 50 a page, without authors, and rate limited. There's no download route and no bulk export. Only validators receive the files. The v2 command for downloading test content, `dataset pull`, is gone.

## What won't work

- **Recognising your own posts.** They aren't in your model's corpus.
- **Flooding one class.** `min(V_pos, V_neg)` is zero.
- **Reposting.** BitFan refuses a near-copy of a post already written.
- **Copying a model.** The same weights have the same fingerprint, and the earliest committer holds it. Renaming files or mirroring the repository changes nothing.
- **Answering `YES` to everything.** Test content is balanced by construction and production content is mostly compliant, so a degenerate model is wrong on most items.
- **A bigger model for a small gain.** Each doubling past 500M parameters costs a point of accuracy, so a larger model has to be that much better to rank level.
- **Tuning someone's model by a hair.** A later model has to beat an earlier one by more than a point to take its place.
