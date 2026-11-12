# The hub contract

The **hub** is a small service the owner team runs, in a separate private repository. Each closed UTC day it combines BitFan's contribution feed with a chain index, signs the result as an **epoch**, and serves it to validators with the day's two content files. It stores validators' signed records, and it publishes the credit ledger and a delayed, paged view of test content.

This page is the contract as the validator client in this repository implements it. The client is `src/prometheon/hub/`, and `prometheon.hub.fake.FakeHub` serves the same contract for tests and for [`localnet.md`](./localnet.md). Where this page and the models in `src/prometheon/hub/models.py` disagree, the models are what runs. Report the difference.

**What the hub is trusted for, and what it isn't.** Validators score what the hub signed: the miner list and member counts, the frozen models with their payment status, and the content files. A validator can't independently check BitFan's identity verification or member counts. The hub can't label content, run models, set weights or alter a validator's record, and everything it signs is public.

---

## Conventions

- **Canonical JSON**: keys sorted at every level, no whitespace, UTF-8 with non-ASCII written as is, and **no floats**. Every instant is an integer. Signatures are always over canonical bytes.
- **Errors**: `{"error": {"code": "db.*", "message": "...", "retry_after_seconds": ?, "available_at": ?}}`. The codes are `db.bad_request`, `db.bad_cursor`, `db.auth_malformed`, `db.auth_invalid`, `db.auth_expired`, `db.auth_replayed`, `db.not_authorized`, `db.embargoed`, `db.not_found`, `db.snapshot_not_ready`, `db.duplicate_evaluation`, `db.invalid_payload`, `db.rate_limited` and `db.internal`.
- **Signed requests** carry `x-prometheon-netuid`, `-hotkey`, `-nonce` (32 hex), `-issued-at`, `-body-sha256` and `-signature`. The signature is the caller's sr25519 hotkey over the canonical envelope under the domain `PROMETHEON_V2_DB_REQUEST`. The hub allows 300 s of clock skew and remembers each nonce for 900 s.

## Routes

| Route | Auth | Returns |
|---|---|---|
| `POST /api/v1/validator/epoch` `{"day":"YYYY-MM-DD"}` | signed by a hotkey with a validator permit | `{"epoch":{...},"urls":{"test":"...","production":"..."},"urls_expire_at":<unix s>}`. URLs last 1 hour. `409 db.snapshot_not_ready` (with `retry_after_seconds`) while the day can still be built; `404 db.not_found` if it never will be |
| `POST /api/v1/validator/weights` `{"day":"YYYY-MM-DD"}` | signed by a hotkey with a validator permit, exactly like the epoch request | `{"record": <the owner validator's record exactly as posted>}`, for validators in copy mode. `409 db.snapshot_not_ready` (with `retry_after_seconds`) until that record exists; `404 db.not_found` for a day that never will have one. The hub serves the owner validator's record only, and the client refuses a record signed by any hotkey other than its `copy_from` |
| `GET /api/v1/epochs/{day}` | public | the epoch without URLs, bare or as `{"epoch":{...}}` |
| `POST /api/v1/results` | the record's own signature; the envelope is optional | `{"evaluation_id","accepted_at","duplicate"}`. The same bytes again: `200` with `duplicate: true`. A different record for the same day and validator: `409 db.duplicate_evaluation`, and the first stands |
| `GET /api/v1/results/{day}/{validator}` | public | the record exactly as it was posted, for anyone auditing a validator. Copy mode doesn't use it |
| `GET /api/v1/corpus?day&cursor` | public, rate limited | test content from day N, available from 00:00 UTC on day N+2. 50 items a page, each only `{id, body, claimed_violating, created_at}`, with no author or miner. Earlier days return `403 db.embargoed` with `available_at`. There's no download route |
| `GET /api/v1/credits/{coldkey_or_hotkey}` | public | the credit balance and ledger, below |

---

## The epoch, `prometheon-epoch/1`

Built at 01:00 UTC on D+1 from the feed for day D and the chain, frozen at the first block at or after 00:00 UTC on D+1.

```jsonc
{
  "schema": "prometheon-epoch/1",
  "netuid": 1,
  "day": "2026-11-05",
  "frozen_block": 123456,
  "scoring_version": "prometheon-scoring/3.1",
  "policy": {"version": "2026-08-07", "sha256": "<sha256 of content_policy.md>"},
  "miners": [
    {"hotkey": "5...", "uid": 3, "coldkey": "5...",
     "verified_active_members": 14, "members": 40, "eligible": true}
  ],
  "models": [
    {"hotkey": "5...", "repo": "org/name", "revision": "<40 hex>", "commit_block": 1,
     "fingerprint": "<64 hex>", "payment": "paid", "parameters": 494032768}
  ],
  "files": {"test": {"sha256": "<64 hex>", "count": 1}, "production": {"sha256": "<64 hex>", "count": 1}},
  "key_id": "hub-2026-11",
  "signature": "0x<128 hex>"
}
```

- **Signature:** Ed25519 over the canonical bytes of the epoch without `signature`, with `key_id` inside the signed bytes. Validators pin the public key, 64 lowercase hex, under its `key_id` in `[hub.epoch_keys]`. Unknown fields are accepted, and they're covered by the signature. A change that matters to scoring changes `schema` or `scoring_version`, and validators refuse those.
- **Content hash:** sha256 of those same signed bytes. Records name it as `epoch_content_hash`.
- **`miners`:** every neuron registered at `frozen_block`, sorted by uid, each with its coldkey.
  - `verified_active_members` and `members` come from BitFan, and are 0 for a neuron with no group. Both are required.
  - Validators refuse an epoch where `verified_active_members > members`.
  - `eligible` is the hub's own view, for its dashboard. Validators decide from the two counts and their config.
- **`models`:** one row per registered hotkey whose frozen commitment decodes and has a fingerprint, sorted by `(commit_block, hotkey)`.
  - `commit_block` is the **first** block at which that hotkey committed that fingerprint. It also decides ranks: among models within a point of the best score, the earliest wins.
  - Validators evaluate `payment == "paid"` only, as published.
  - `parameters` is optional: `P` as the hub counted it, by the definition below. Validators count for themselves and use their own count. A differing count is noted in their log.
- **`policy`:** sha256 of the bytes of `content_policy.md`, and its `Version:` line. A validator with a different policy file stops.
- **`files`:** sha256 over the gzip bytes, and the line count. A mismatch stops the cycle.

### Weights fingerprint

```text
fingerprint = sha256( "\n".join( sorted( lfs_sha256 of every *.safetensors file ) ) )
```

Lowercase hex, no trailing newline. The LFS sha256s come from `GET /api/models/{repo}/revision/{revision}?blobs=true`. File names don't count. A repository with no safetensors, or with one outside LFS, gets no fingerprint, isn't listed and isn't charged.

### Parameter count

`P` is the sum of the element counts of every tensor in every `*.safetensors` file at the revision. A tensor's element count is the product of its `shape`, and a scalar (`shape: []`) counts 1. `__metadata__` isn't a tensor.

It's read from each file's header with two HTTP range requests on `/{repo}/resolve/{revision}/{file}`: bytes 0 to 7 give the header length `N`, little-endian, and bytes 8 to `8 + N - 1` are the JSON header. Nothing else in the header is checked. A file whose header can't be read makes the model invalid: `N` of 0, `N` over 8 MiB, `8 + N` past the file's size, a header that isn't a JSON object of tensors with lists of non-negative integer shapes, or a weight file that isn't at the top of the repository.

### Payment

The hub lists a model only if it passes the hub's own copy of the registry checks: the file manifest, at most 256 weight files, `config.json` with a `model_type` the runtime knows, the 16 GiB and 64 MiB ceilings, weights in LFS, and `P` ≤ 4,000,000,000. For each fingerprint, among the hotkeys committing it in this epoch:

1. A hotkey that has already spent a credit on this fingerprint holds it: `paid`, with no new spend.
2. Otherwise, in `(commit_block, hotkey)` order, the first hotkey whose coldkey has a free credit at `frozen_block` holds it: `paid`. **The credit is spent at this freeze only if the holder's miner is eligible**, which is the only case in which validators evaluate it. Otherwise it is set aside for this freeze and carried forward.
3. Everyone else is `duplicate` if a holder exists, or `unpaid` if nobody could pay. Neither spends anything.

So a credit is spent only when a model is evaluated. Anyone can recompute the spends from the published epochs alone: a spend happens when a row is `paid`, its miner's row is eligible, and the pair has no earlier spend. An unpaid commit doesn't claim a fingerprint. Deposits count toward a freeze only with `block <= frozen_block`.

**Validators and the hub must agree on the model limits.** If a validator ran different size ceilings or a different `model_type` list from the hub, a model could be charged but not evaluated there, or the reverse. Keep `[evaluation]` at the shipped values.

---

## Feed files

`test.jsonl.gz` and `production.jsonl.gz` are BitFan's bytes, unchanged, one JSON object per line. A line that breaks a rule below is dropped for that day, on every validator alike, and counted in the operator's log. The file itself must still match its sha256 and line count. **If every test line in a non-empty file is rejected, validators stop instead of burning the data pool.**

**Test line**, every field required:

```json
{"id":"t_...","miner_hotkey":"5...","author":"<per-day pseudonym>","claimed_violating":true,
 "body":"...","normalized_sha256":"<64 hex>","created_at":"2026-11-05T12:34:56.789Z","verified":true}
```

- `verified` must be JSON `true`. It's BitFan's marker that the author passed identity verification.
- `miner_hotkey` must be listed in the epoch's `miners`.
- `claimed_violating` must be a JSON boolean.
- `author` must be non-empty.
- `body` must be non-empty, at most 32,000 characters.
- Other fields are ignored.

**Production line:** `{"id","body","created_at"}`, plus an optional `verified: true` and nothing else. BitFan samples about 200 public posts a day into this file. A production line carrying `author`, `miner_hotkey` or any other field is rejected, because production content is unattributed.

**Both:**

- `created_at` is RFC 3339 with `Z` or a numeric offset, optionally with a fraction, and falls inside UTC day D.
- `id` is at most 128 characters and unique across both files. A repeat is rejected; the first occurrence stands.

### `normalized_sha256`

sha256 (lowercase hex) of the UTF-8 text after, in order:

1. Unicode NFKC
2. lower case
3. each run of tab, LF, VT, FF, CR, space, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F or U+3000 becomes one space
4. delete every character in general category P, S, Cf or Cc, and U+FE00 to U+FE0F
5. collapse runs of spaces to one, and trim

| Input | Normalised | sha256 |
|---|---|---|
| `Hello,   WORLD!!!` | `hello world` | `b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9` |
| `h.e.l.l.o` | `hello` | `2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824` |
| `!!!` | empty | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

A missing or malformed value rejects the line. A well-formed value that disagrees with the validator's own computation doesn't reject it: the validator logs the count. Validators no longer use the value for scoring. BitFan refuses a near-copy when it's written, and the field stays so the feed format doesn't change.

---

## The results record

The body of `POST /api/v1/results` is the validator's signed `EvaluationSubmission`. It's signed with the validator's hotkey under `PROMETHEON_V2_EVALUATION`, over every top-level field except `signature`. A validator signs it once and keeps the bytes, so a retry re-posts the identical record.

- `date`, `netuid`, `validator_hotkey`, `scoring_version`, `policy_version`
- `epoch_content_hash`: the epoch this record scored
- `corpus_content_hash`, `labelled_test_count`, `labelled_production_count`
- `results[]`, one row for every eligible miner and every hotkey with a listed model:
  - `dataset_submitted_count` (`S`), `dataset_violating_confirmed` (`V_pos`), `dataset_compliant_confirmed` (`V_neg`), `dataset_valid_count` (`V_pos + V_neg`), `dataset_score_micro`
  - `model_status`: `not_committed`, `unpaid`, `duplicate`, `ineligible`, `invalid`, `unreachable`, `timed_out` or `evaluated`
  - `model_items_scored`, `model_correct_count` and `raw_accuracy_bp`
  - `model_parameters`: `P` as this validator read it, or `null` when the model never reached that check
  - `size_penalty_bp`: `floor(100 × log2(P / 500,000,000))`, 0 up to 500M, for a model that passed every check, or `null`
  - `model_score_bp`: `max(0, raw_accuracy_bp − size_penalty_bp)`, and 0 for a model with no score, such as one that timed out
  - `model_rank`: its place, or `null` when it doesn't rank
  - `weight`: the units that hotkey earned, never including burn
- `burned_weight`: everything sent to the owner hotkey, including units of a hotkey that deregistered before the send. Rows plus `burned_weight` sum to exactly 1,000,000 and reproduce the vector the validator sent.
- `started_at`, `completed_at`, `signature`

## Credits, `GET /api/v1/credits/{coldkey_or_hotkey}`

```json
{
  "coldkey": "5...",
  "balance": 2,
  "remainder_alpha_rao": 12500000000,
  "submission_fee_alpha_rao": 50000000000,
  "entries": [
    {"kind": "deposit", "block": 8379737, "extrinsic_hash": "0x...", "alpha_rao": 60000000000, "credits": 1},
    {"kind": "spend", "block": 8380000, "credits": -1, "hotkey": "5...", "fingerprint": "<64 hex>"}
  ]
}
```

- A hotkey resolves to its coldkey. An unknown key answers `200` with `balance: 0`.
- Amounts are integer rao.
- Entries are newest first, and capped at 500 with `truncated: true`.
- An `unresolved_deposit` entry is a transfer whose alpha amount couldn't be established. It has `credits: 0` and is settled by a person.
- Only `transfer_stake` calls to the treasury coldkey with origin and destination netuid both equal to the subnet's count. The ledger reads the call's `alpha_amount`, because the event reports a TAO equivalent.
