# Contributing

## The gate

Everything below must pass before a change lands. CI runs the same commands.

```bash
uv sync --group dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/prometheon
uv run pytest
```

`mypy` runs `--strict`. New code is fully annotated; there is no opt-out list to
add a module to.

---

## Tests

Three markers, and which one a test gets is a real decision:

| Marker | Means |
|---|---|
| `unit` | no network, no chain, no clock |
| `contract` | pins a wire format or a published constant against a fixture |
| `integration` | exercises composition across modules |

### Mocks must not agree with themselves

This is the rule that matters most here, because breaking it has already shipped
real defects in this codebase.

A mock that returns what the code under test expects proves the code is
self-consistent, not that it is correct. Two concrete failures, both caught only
after being written the right way:

- A contract test built a metagraph object by hand and passed it straight to the
  transposer, never calling the adapter function that fetches one. It pinned a
  type the adapter never receives, and passed while the real path returned an
  **empty snapshot**, with every miner reading as deregistered.
- A test helper silently ignored an unknown field name, so a test asserting on
  `extrinsic_id`, a field the SDK does not have, passed by falling through to
  the status message instead. It hid the fact that the adapter could never
  produce a real receipt.

So:

- **Drive HTTP through `httpx.MockTransport`**, over real request and response
  objects. Never monkeypatch a client method.
- **Drive the SDK through real SDK objects** where they can be built offline.
  `tests/contract/test_bittensor_sdk_shape.py` does this, and goes *through* the
  adapter rather than around it.
- **Make fakes strict.** A fake that accepts a field the real thing does not have
  is worse than no fake.
- **Prefer the shipped fake.** `FakeDbLayer` is a complete implementation of the
  `/v2` contract; use it rather than stubbing the client.

### What a test should be named after

The behaviour, and preferably the consequence of it being wrong:

```python
def test_a_dead_model_does_not_take_emission_off_the_honest_field() -> None:
def test_an_unsynced_metagraph_does_not_read_as_a_deregistered_field() -> None:
```

not `test_allocate_model_pool_2`.

---

## Comments

Comment the **why**, never the what. The code says what it does.

Things that belong in a comment:

- why a constant has the value it has
- what breaks if a check is removed
- which failure mode a piece of defensive code exists for
- a bug that was here before, so it does not come back

That last one is a convention here. Several modules carry a paragraph describing
a defect that shipped, why it was invisible, and what now prevents it. Those
paragraphs are the most valuable comments in the codebase. Please add to them
rather than trimming them.

---

## Changing anything that moves emissions

Every reward-affecting constant lives in `ScoringConfig`
([`src/prometheon/config.py`](src/prometheon/config.py)) with the reasoning that
produced it. A change to one of them:

1. changes the constant **and** its comment,
2. updates [`docs/scoring.md`](docs/scoring.md), which is the published rule,
3. adds a test pinning the new behaviour,
4. bumps `SCORING_VERSION` in `prometheon/cli/cycle.py` if two validators on
   different builds would now compute different weights.

Point 4 is not optional. The version is published in every signed evaluation
record so a divergence can be attributed to a version skew rather than to a
disagreement about the data.

If you change the canonical engine or wrapper, you are invalidating every
deployed model on the subnet. Read the migration policy in
hash to `LEGACY_WRAPPER_HASHES` for the window.

---

## Determinism

Two validators with identical inputs must compute an identical weight vector.
That constrains ordinary-looking code:

- **No floats in anything signed or hashed.** Canonical JSON rejects them.
  Ratios travel as basis points or fixed-point micros. Scoring computes on
  `Fraction`.
- **Every sort needs a total order.** Sorting on score alone is not deterministic
  when scores tie, and a tie between rank 1 and rank 2 is 1500bp of emission.
  Break ties down to hotkey.
- **Never iterate a set or a dict for anything that reaches a result.** Build
  ordering from the data, not from insertion order.
- **No wall-clock reads inside scoring.** Time arrives as a parameter.

---

## Commits and pull requests

Conventional commits: `fix(chain):`, `feat(scoring):`, `docs:`, `test(registry):`.

A PR body says what changed and why. If it fixes a defect, describe the failure
mode. That text usually belongs in a code comment too.

Commits must be signed; CI verifies this through the GitHub API rather than
`git log --pretty=%G?`, which reports unsigned for correctly signed commits when
the runner has no `allowed_signers` file.
