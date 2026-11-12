<p align="center">
  <img src="Prometheon_Logo.svg" width="120" alt="Prometheon" />
</p>

# Prometheon

Prometheon is a Bittensor subnet whose goal is an **AI moderation agent**. It starts with text content moderation: a model that reads a post and decides whether it breaks a written policy. It is built for platforms, communities and businesses that need moderation.

> **Status: pre-launch.** The subnet is not registered yet. The team will register it and announce the netuid then. Until that announcement, nothing in this repository refers to a live subnet: the netuid, the chain endpoint and the treasury coldkey are required settings with no default, and every end-to-end test runs on a local chain ([`docs/localnet.md`](./docs/localnet.md)).

---

## What the subnet does

Miners earn in two ways, and every validator scores both from the same signed daily record:

1. **Data, 70% of emission.** A miner leads a group on [BitFan](https://bitfan.ai), a separate consumer product and Prometheon's first data partner. Identity-verified members of the group write test posts and say whether each one breaks the policy. Validators label every post themselves and pay for the claims their labeller confirms.
2. **Models, 30% of emission.** A miner publishes moderation model weights on Hugging Face and commits the repository and revision on chain. Validators download every eligible model whose fee is paid, run it on their own hardware over the day's content, and pay the three best: 25%, 4% and 1% of emission.

```text
BitFan ──daily feed──▶ hub ──signed epoch──▶ validators ──▶ label, run every model, score
                        ▲                         │
          chain index ──┘                         └──▶ set_weights, publish a signed record
```

The **hub** is a small service the owner team runs. Each day it combines BitFan's feed with what the chain says (registered neurons, frozen model commitments, credit payments), signs the result, and hands it to validators. Validators check that signature against a key pinned in their config, then do the rest themselves: labelling, running models, scoring and setting weights.

BitFan and Prometheon are separate products. They share no database, no login and no code. Other data partners can be added later on the same feed format.

### Roadmap

1. **Now:** text moderation against one policy, [`content_policy.md`](./content_policy.md).
2. Multiple policies, and image and video moderation.
3. Bot detection, fraud detection and account-farming detection.
4. The agent: moderating platforms, Discord, Telegram and other social traffic end to end.

**The reward split moves with the roadmap.** It starts at 70% data and 30% models, because inviting real people and getting them to write test content costs far more today than training a text model. As user numbers grow and the model requirements get harder, the team expects the share for model training and agent development to reach 80 to 90%. Any change ships as a new scoring version that every validator adopts together. It is never a quiet config edit.

---

## The rules, exactly

These match the code under `scoring_version = "prometheon-scoring/3.1"`. [`docs/scoring.md`](./docs/scoring.md) has the arithmetic with worked examples.

| Rule | Value |
|---|---|
| Data pool | 70% of emission, split among the top 10 eligible miners in proportion to their data score |
| Data score | `2 × min(V_pos, V_neg) × (V_pos + V_neg) / S` |
| Model pool | 25%, 4% and 1% of emission to the top three models |
| Model size | At most 4,000,000,000 parameters and 16 GiB of weights. Past 500 million parameters, accuracy is reduced by one point per doubling: `floor(100 × log2(P / 500M))` basis points |
| Model ranking | Score is accuracy less the size penalty. Each place goes to the model committed first among those within one point of the best score |
| Time limit | 30 minutes per model on each validator. A model that hasn't finished gets no score from that validator that day |
| Unfilled places | An unfilled model place goes to the lowest-ranked winner present: two models get 25% and 5%, and one model gets 30%. A pool burns only when nobody qualifies for it |
| Eligible miner | At least **10** identity-verified active members **and** at least **30** members in total, as counted in the signed epoch. Both pools. Every member is phone-verified, and the 30 includes the 10 |
| Whose content counts | Only identity-verified people on BitFan (Stripe Identity). See below |
| Model fee | 50 alpha per new model, paid in advance as a credit. No refunds |
| Burn | Anything nobody qualifies for is weighted to the subnet owner hotkey |

`S` is a miner's test posts for the day, as the feed carries them. BitFan refuses a near-copy of an earlier post when it's written, so validators don't remove copies themselves. `V_pos` counts posts claimed violating that the validator's labeller confirmed violating. `V_neg` counts posts claimed compliant that it confirmed compliant. Because the score uses `min(V_pos, V_neg)`, a flood of one class earns nothing.

### Whose content counts

Only content from people who passed identity verification (Stripe Identity) on BitFan. BitFan's export includes only human accounts at that level that aren't banned or on hold, in groups whose leader wallet is a registered, unbanned hotkey. BitFan also limits each person to 5 violating and 5 compliant test posts per UTC day. Validators refuse any test line in the feed that doesn't carry `"verified": true`.

Be clear about what a validator can and can't check. It can check that the feed is the one the hub signed and that every line is well formed. **It can't check identity verification itself, or the member counts.** Those come from BitFan through the hub. They are published with every epoch, so anyone can compare them afterwards.

### Model submissions and credits

A miner pays 50 alpha of stake to the treasury coldkey on the subnet (`prometheon credits buy` sends the `transfer_stake`) and gets one credit. A credit is spent only when a model is evaluated: at the first freeze where its fee is covered, its miner is eligible and it passes the model checks. Re-committing the same weights costs nothing. A model without a credit is listed as `unpaid` and isn't scored. It's picked up at the next freeze after a credit arrives. Credits belong to the coldkey that paid, and are shared by every hotkey under it. **There are no refunds.** The ledger is public, with the block number and extrinsic hash of every entry.

### How the owner team takes part

- **The owner runs a validator like everyone else, with this public CLI.** It uses no private build and gets no special inputs.
- **The owner runs the hub**, which signs each day's epoch. That signature is a trust point: validators score what the hub signed. The hub can't label content, run models or set anyone's weights, and everything it signs is published.
- **Burned emission is weighted to the subnet owner hotkey.** That happens only when nobody qualifies for a pool, or for the units of a miner that deregistered before weights were sent.
- **The treasury coldkey**, set at launch, receives model fees.

---

## Installation

Requires Python `>=3.10,<3.15` and a `btcli`-managed Bittensor wallet.

```bash
git clone https://github.com/BitSpaceorganization/prometheon_v2
cd prometheon_v2
uv sync                    # miners, and validators in endpoint mode
uv sync --extra wrapper    # validators that run models (the default, local mode)
```

`--extra wrapper` installs torch and transformers, which run the models. Without them, a local-mode validator that has a model to run stops at a pre-flight check before it labels anything.

---

## Miners

```bash
cp configs/finney.example.toml ~/prometheon.toml
# fill in: [chain] netuid, [wallet], [hub], [credits] treasury_coldkey, [miner]

uv run prometheon credits balance --config ~/prometheon.toml
uv run prometheon credits buy --config ~/prometheon.toml --dry-run    # then without --dry-run
uv run prometheon model commit --config ~/prometheon.toml
uv run prometheon model verify --config ~/prometheon.toml
```

Your Hugging Face repository holds **weights, config and tokenizer only, with no executable code**. Read [`docs/miner.md`](./docs/miner.md) before you pay for a credit.

## Validators

```bash
cp configs/finney.example.toml ~/prometheon.toml
# fill in: [chain] netuid, [wallet], [hub] base_url and epoch key, [labelling]
export OPENAI_API_KEY="..."     # you pay for ground-truth labelling

uv run prometheon validator run --config ~/prometheon.toml    # one day; dry_run = true first
```

A validator scores yesterday once a day at 04:00 UTC, and re-posts its weights every 30 minutes. Scoring the day needs one GPU with 24 GB, for example an RTX 4090. Endpoint mode needs none, but it copies another validator's opinion. Read [`docs/validator.md`](./docs/validator.md) before choosing.

---

## Repository layout

```text
src/prometheon/hub/         the hub client: signed epochs, feed files, results, credits; a fake hub
src/prometheon/chain/       metagraph, commitments, weight submission, stake transfers
src/prometheon/registry/    model checks against Hugging Face: manifest, size, fingerprint, parameters, duplicates
src/prometheon/labelling/   ground-truth labelling, one post per request
src/prometheon/evaluation/  corpus assembly and the forced-choice model engine
src/prometheon/scoring/     data and model scores, the size penalty, the split, burn
src/prometheon/security/    canonical JSON and signatures
src/prometheon/cli/         the miner and validator commands
tools/localnet/             a whole cycle on a local chain, with real extrinsics
content_policy.md           the authority for every verdict
configs/                    example configuration
docs/                       guides, the scoring rules and the hub contract
```

## Documentation

| Document | Purpose |
|---|---|
| [`content_policy.md`](./content_policy.md) | The authority for every verdict |
| [`docs/scoring.md`](./docs/scoring.md) | How emissions are computed, with worked examples |
| [`docs/miner.md`](./docs/miner.md) | Earning from data and from models; credits |
| [`docs/validator.md`](./docs/validator.md) | Running the daily cycle |
| [`docs/hub-contract.md`](./docs/hub-contract.md) | The signed epoch, the feed lines, and the hub's routes |
| [`docs/localnet.md`](./docs/localnet.md) | Running the whole cycle on a local chain |
| [`docs/open-design-issues.md`](./docs/open-design-issues.md) | The two design questions from the v2 review, the time budget and the efficiency term, and how 3.1 settled them |

## License

[MIT](./LICENSE). © 2026 BitSpace.

## Reporting vulnerabilities

See [`SECURITY.md`](./SECURITY.md).
