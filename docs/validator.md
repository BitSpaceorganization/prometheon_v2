# Validating

A validator runs **one cycle per day**, scoring the previous day at 04:00 UTC.

You pay for two things: ground-truth labelling (an OpenAI-compatible key) and
model invocation (a subnet-issued Chutes key). Miners pay for their own GPUs.

---

## Setup

```bash
git clone https://github.com/BitSpaceorganization/prometheon_v2
cd prometheon_v2
uv sync

cp configs/mainnet.example.toml ~/prometheon-mainnet.toml
$EDITOR ~/prometheon-mainnet.toml        # set [wallet] name and hotkey

export OPENAI_API_KEY="…"     # your own key — you pay for ground-truth labelling
export CHUTES_API_KEY="…"     # your own Chutes key, authorised on the subnet by the owner
```

**The Chutes key is your own, not a shared one.** Create a Chutes account, then
ask the subnet owner to authorise it for netuid 108. That grant carries the
*invoke private chutes of a subnet* role, which is what lets you call miners'
deployments — every miner's chute is private, and without the role the platform
answers `404` for chutes that plainly exist. Miners do not share anything with
you individually, and no miner ever hands over a credential.

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
4. verify each model              ← Hugging Face + Chutes, independently
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
chain and then verified against Hugging Face and Chutes independently. The two
are compared, and a disagreement is recorded in the published result. The chain
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

What does **not** stop a cycle: any single miner. A model that times out,
returns garbage, declares a false `Content-Encoding`, floods the response, or
refuses the connection is scored incorrect for that batch and the cycle
continues. That is a promise the code keeps. Every validator evaluates the same
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

## Running it daily

```cron
0 4 * * * cd /opt/prometheon_v2 && /usr/local/bin/uv run prometheon validator run \
    --config /etc/prometheon/testnet.toml >> /var/log/prometheon.log 2>&1
```

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

Evaluation cost is per invocation against miners' chutes, using the
subnet-issued key. You do not pay for their GPU time.
