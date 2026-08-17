<p align="center">
  <img src="Prometheon_Logo.svg" width="120" alt="Prometheon" />
</p>

# Prometheon V2

A Bittensor subnet that develops, benchmarks, and continuously improves **open-source content-moderation models**.

> **Phase scope.** This repository is Phase 2 and replaces the Phase 1 activity-reward mechanism entirely. Phase 1 lives in `prometheon_v1` and is no longer the reward path.

---

## What this subnet does

Miners compete to build the best moderation model, and they supply the data it is measured on.

1. **A miner grows a BitFan Fan Group.** Their qualified users write posts and replies they believe violate the platform's content policy, marked as **test content**. Those are recorded, never published.
2. **The subnet collects a day of content**: that test content, plus a sample of real production content.
3. **Validators label it** against [`content_policy.md`](./content_policy.md), producing the day's ground truth.
4. **Miners publish a moderation model** to Hugging Face and deploy it to Chutes behind a canonical, hash-pinned wrapper, committing `(hf_repo, revision_sha, chute_id)` on chain.
5. **Validators run every model** over the labelled corpus and set one weight vector.

Two things are rewarded, in equal measure: **contributing usable evaluation data**, and **building the model that judges it best**.

```text
BitFan users write test content   →   subnet DB layer   →   validators label it
                                                              ↓
miners publish models to HF  →  Chutes  →  validators evaluate  →  set_weights
```

---

## What makes the measurement trustworthy

Code in this repository enforces each of these. There is no rule anyone has to remember to follow.

**A miner is never scored on content its own users wrote.** Test items carry their contributing miner, and each model's accuracy is computed only over other miners' items. Watermarking your own submissions buys nothing.

Miners also cannot see the corpus they are measured on. A day's test content is released to miners two days later, after the evaluation that used it has finished. Memorisation is not a strategy.

Every model runs identical code. Miners supply weights and nothing else. The inference path is one subnet-authored script, hash-pinned, running inside one subnet-built container image — and validators check both from outside, against the deployment itself. A hash proves which bytes are on disk; pinning the image is what makes them run somewhere the miner did not assemble. A score difference is a model difference.

**Verdicts cannot be malformed, and a model cannot refuse.** Decoding is forced-choice: the engine compares probability mass on `YES` against `NO` and takes the larger. Nothing is generated freely, so there is no output to parse and no refusal to interpret.

The corpus contains text written to violate policy, some of it written to manipulate a classifier. Items are therefore judged one at a time, never concatenated, so an instruction buried in one post cannot reach the verdict on another.

Validators verify eligibility themselves, from chain commitments, Hugging Face, and Chutes. The subnet's own database is not trusted for it.

---

## Repository layout

```text
src/prometheon/canonical/   the engine and wrapper every miner deploys, and their integrity checks
src/prometheon/chain/       commitments, metagraph, weight submission
src/prometheon/registry/    model eligibility: manifest, hashes, provenance, duplicates
src/prometheon/dbclient/    the /v2 client, its contract, and an in-memory fake
src/prometheon/labelling/   ground-truth labelling with injection-hardened batching
src/prometheon/evaluation/  corpus assembly and model execution
src/prometheon/scoring/     dataset and model rewards, allocation, burn
src/prometheon/cli/         the miner and validator command surface
content_policy.md           the authority for every verdict
configs/                    pre-pinned example configuration
docs/                       operator guides and the contracts other teams build against
```

---

## Installation

Requires Python `>=3.10,<3.15` and a `btcli`-managed Bittensor wallet.

```bash
git clone https://github.com/BitSpaceorganization/prometheon_v2
cd prometheon_v2
uv sync
```

Every command below assumes `uv run` in front of `prometheon`. To drop the prefix, `source .venv/bin/activate` once.

---

## Miners

You need a Fan Group with **at least 500 qualified users**, meaning active real accounts that have cleared the platform's anti-farming checks. That gate applies to both reward streams.

Register your Fan Group and link its leader account to your hotkey at **[bitfan.ai/me/prometheon](https://bitfan.ai/me/prometheon)**. That is a platform step, not a subnet command, and nothing else works until it is done.

```bash
# 1. Render the canonical wrapper to deploy on Chutes
uv run prometheon model render --config ~/prometheon-testnet.toml \
    --hf-repo <you>/<model> --hf-revision <40-char-sha> \
    --chutes-user <you> --chute-id <uuid> --output wrapper.py

# 2. Commit it on chain — this is the submission
uv run prometheon model commit --config ~/prometheon-testnet.toml

# 3. Check what a validator will conclude about your deployment
uv run prometheon model verify --config ~/prometheon-testnet.toml

# 4. Pull released datasets to train against
uv run prometheon dataset pull --config ~/prometheon-testnet.toml --date 2026-08-05
```

Your Hugging Face repository holds **weights, config and tokenizer — no executable code at all**. Any other file and your deployment is not scored. See [`docs/miner.md`](./docs/miner.md).

**Authorise the subnet to call your model** with `chutes share`. You keep your credential, and no key is ever handed to a validator. You are the chute owner, so you pay for your own GPU hours.

---

## Validators

A validator runs **one cycle per day**, starting at 04:00 UTC against the previous day's data.

```bash
cp configs/testnet.example.toml ~/prometheon-testnet.toml   # then edit [wallet]
export OPENAI_API_KEY="…"        # you pay for ground-truth labelling
export CHUTES_API_KEY="…"        # the subnet-issued key for calling miners' models

uv run prometheon validator run --config ~/prometheon-testnet.toml
```

Each stage is runnable on its own for debugging. See [`docs/validator.md`](./docs/validator.md).

---

## Documentation

| Document | Purpose |
|---|---|
| [`content_policy.md`](./content_policy.md) | The authority for every verdict |
| [`docs/miner.md`](./docs/miner.md) | Publishing, deploying, and committing a model |
| [`docs/validator.md`](./docs/validator.md) | Running the daily cycle |
| [`docs/canonical-wrapper.md`](./docs/canonical-wrapper.md) | The contract every model is built against |
| [`docs/db-api-contract.md`](./docs/db-api-contract.md) | The `/v2` interface the subnet DB layer implements |
| [`docs/scoring.md`](./docs/scoring.md) | How emissions are computed |

---

## Status

Phase 2 targets Bittensor **testnet netuid 481**.

## License

[MIT](./LICENSE). © 2026 BitSpace.

## Reporting vulnerabilities

See [`SECURITY.md`](./SECURITY.md).
