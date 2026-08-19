# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/BitSpaceorganization/prometheon_v2/security/advisories/new).

Please do **not** open a public issue for anything that lets a miner earn
emission they did not earn, or lets anyone influence what a validator computes.
Those are worth real money on a live subnet, and a public issue is a starting
pistol.

Include what you would want to receive: what you did, what happened, and what
you expected. A failing test is the most useful thing you can send.

We aim to acknowledge within 72 hours.

---

## What counts as a vulnerability here

This is a subnet, so the interesting attacks are economic. Anything in these
categories is in scope:

**Earning without competing.** A way for a miner to score without a working
model: forging verdicts, influencing its own corpus, getting credit for another
miner's model, or making a rival's model score worse than it should.

**Influencing ground truth.** Anything that changes what the labeller concludes,
including content crafted to subvert a batch.

**Making validators disagree.** Two honest validators with the same inputs must
compute the same weight vector. Non-determinism, iteration order, a float in a
signed payload: anything that makes them diverge costs honest validators vtrust.

**Taking a validator down.** Anything one miner can do that stops a cycle
completing. Because every validator runs the same cycle at the same time against
the same miners, a single crashing input is a subnet-wide outage rather than one
operator's problem.

**Escaping the evaluation contract.** Any way for a scored model to differ from
the checkpoint the miner committed, or for a repository to get code executed on
a validator's machine. Validators download and run miner-supplied weights, so
the file manifest is a security boundary, not housekeeping.

**Credential exposure.** Any path that leaks a validator's OpenAI key.

### Not in scope

- A model that is slow, large, or unhelpful. Those cost the miner score, which
  is the mechanism working.
- Rate limits or availability of Hugging Face or the DB layer.
- Anything requiring a validator to run a modified build. A validator that
  changes its own scoring diverges and loses vtrust, which is the design working.
- Findings that a model gives a wrong verdict. That is what the competition is
  for.

---

## The properties this codebase is trying to hold

Stated so you know what to attack, and so a report can name which one broke.

**A score difference is a model difference.** Every model runs identical,
hash-pinned code. Miners supply weights and nothing else.

**A miner is never scored on content its own users wrote.** Self-exclusion is
enforced in code, not by a rule.

**Miners cannot see the corpus they are measured on.** Two-day embargo.

Ground truth is fixed before any model sees the corpus, so no model can
influence what it is measured against.

**Items are never concatenated.** One item per decision, with the policy KV-cache
copied per item so one item's activations cannot reach another's verdict.

A validator verifies everything itself. Model eligibility is computed from the
chain and Hugging Face. It is never taken on trust from the subnet's own
database, which is a service this code audits rather than believes.

**No failure reads as success.** Each of these is distinguishable from the
success it resembles: a rejected extrinsic, an unusable labelling response, a
model that answered nothing. Several past defects in this repository were exactly
this mistake, and their fixes carry comments saying so.

---

## Handling of secrets

Validators hold one credential, an OpenAI-compatible key, read from a named
environment variable and never written to logs, error messages, or the published
record. Miners hold none at all: a submission is a public repository and a commit
SHA, so there is no credential for either side to exchange or leak.

There are no API keys for the subnet DB layer. Every request is signed with the
caller's hotkey and checked against the metagraph, so access follows registration
automatically and there is nothing to revoke.

Wallet hotkeys are opened through `bittensor-wallet` and only ever produce a
public ss58 address for display.
