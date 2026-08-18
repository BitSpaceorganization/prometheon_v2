# The canonical wrapper

Every model on this subnet runs identical inference code. Miners supply weights;
the code is fixed, hash-pinned, and verified before a model serves a single
request.

The competition rests on one property:

> **A score difference is a model difference.**

Without it, a miner could win by writing better prompt scaffolding, by
post-processing verdicts, or by detecting evaluation traffic and behaving
differently for it. None of that is available here. The only thing a miner
controls is the weights.

---

## What is pinned

Two artefacts, hashed separately because they are verified in different places:

| Artefact | Lives in | Checked against |
|---|---|---|
| **The engine** (`miner.py`) | the miner's Hugging Face repo | its SHA-256, fetched from the repo |
| **The wrapper** (`wrapper.py`) | the miner's Chutes deployment | its normalised hash, fetched from Chutes |

```bash
uv run prometheon canonical    # prints both
```

---

## The values a miner sets

```python
PROMETHEON_HF_REPO     = "you/model"      # --hf-repo
PROMETHEON_HF_REVISION = "<40-char SHA>"  # --hf-revision
PROMETHEON_CHUTES_USER = "you"            # --chutes-user
PROMETHEON_HOTKEY      = "<ss58>"         # from [wallet]; names the chute
PROMETHEON_GPU_COUNT   = 1                # --gpu-count (1-8)
PROMETHEON_MIN_VRAM_GB = 16               # --min-vram-gb (16-140)
```

Nothing else. Not a constant, not a comment, not a blank line — and in
particular not `tee=True` or the `pro_6000` pool, which are fixed.

### How that is enforced without punishing whitespace

A byte-for-byte comparison would reject a wrapper that differs only in
formatting, which is a bad failure: the miner's code is *identical* in behaviour
and they lose a day to a trailing newline.

Instead the source is **normalised** before hashing:

1. The six values above are replaced with a fixed placeholder, so two miners who
   differ only in model, account, hotkey, or instance size hash identically.
2. The result is parsed and re-emitted from its abstract syntax tree.

Comments, whitespace, and quote style vanish. Anything that changes *behaviour*
survives and changes the hash.

The declared values are then read back out of the AST and compared against what
the miner committed on chain. A regular expression is not used for that read; a
crafted comment could fool one.

---

## The file manifest

A Hugging Face repository may contain **only** these:

```text
.gitattributes            config.json               tokenizer.json
.gitignore                generation_config.json    tokenizer.model
README.md                 merges.txt                tokenizer_config.json
vocab.json
model.safetensors         model.safetensors.index.json
special_tokens_map.json
```

plus sharded weights matching `model-*.safetensors`.

Any other file and the deployment is rejected. The rule is "accept only what we
asked for", rather than trying to reject whatever could hurt us. A repository is
attacker-controlled, and an allowlist is the only version of this check that
does not need updating every time somebody invents a new way to smuggle code
into a model load.

The **image is the subnet's**, referenced by id. It is not built on your machine
and there is no `chute_config.yml`: the Chutes API exposes an image's identity
and never its contents, so an image you built is one no validator can verify.

The **deployment target is fixed and part of the hash**: `tee=True` (Chutes
serves only confidential-compute chutes, and the subnet requires it) and
`include=["pro_6000"]` (the RTX PRO 6000 pool, where TEE capacity is available).
You size the instance with `--gpu-count` and `--min-vram-gb` and you pay for the
GPU hours, but the pool and the enclave requirement are not yours to change.

---

## Why the framing is fixed

**No chat template.** The engine uses a fixed plain-text prompt rather than each
model's `tokenizer.apply_chat_template`. A chat template comes from the miner's
own `tokenizer_config.json`. Using it would let a miner inject text into the
framing of every item their model judges, and would mean no two models saw the
same prompt. Fixed framing costs instruction-tuned models a little headroom, and
costs it to all of them equally.

**Forced-choice decoding.** The engine compares the probability mass on
`{YES, Yes, yes, …}` against `{NO, No, no, …}` at a single position and takes the
larger. There is no free generation, which means:

- nothing to parse, so no parser to disagree about
- no refusal, because there is no text to refuse with
- near-determinism: the same weights and the same input give the same verdict

**One item at a time.** The corpus contains content written to violate policy,
some of it written to manipulate a classifier. Items are never concatenated, so
an instruction buried in one post cannot reach the verdict on another.

**The policy KV-cache is copied per item.** The policy prefix is encoded once and
reused, which is a large saving. But a forward pass *mutates* the cache it is
given. Reusing it without copying would let item 2 attend to item 1's keys and
values, reintroducing exactly the cross-item contamination that per-item
isolation exists to prevent. The failure is subtle and would have been invisible
in testing.

---

## Limits

| Limit | Value | Why |
|---|---|---|
| Policy tokens | 4,096 | the prefix is sent with every batch |
| Content tokens per item | 2,048 | bounds a single hostile item |
| Items per request | 100 | bounds one batch |

**Truncation happens at ingest, not in the wrapper.** If the wrapper truncated,
the labeller would have judged the *full* text while every model read a
*truncated* version. That is an irreducible error in the corpus, one no model
could overcome and one that would look like model error. Ingest truncation means
the labeller and every model see exactly the same bytes.

---

## Version migration

Validators accept a **set** of hashes, never a single one. A bump with no
overlap would invalidate every deployment in the subnet at the same instant.

**Standard bump: 7 days.** The new hash is published, both score normally for a
week, then the old one is removed. No penalty for either during the window.

**Emergency bump: 48 hours.** Reserved for a defect in the wrapper itself: a
prompt-injection escape, or a path that lets a verdict be forged. Some miners
will drop out, and that is the accepted cost of closing a live hole.

Enforcement needs no new machinery. A validator that does not pull the
repository rejects deployments the rest of the field accepts, diverges from
consensus, and loses vtrust. That is the same mechanism that governs policy
updates.

---

## Verifying it yourself

```bash
uv run prometheon model verify --config ~/prometheon-mainnet.toml
```

Runs the validator's own pipeline, not a miner-side reimplementation of it. A
second definition of eligibility could agree with the first for months and then
diverge on the day it mattered.
