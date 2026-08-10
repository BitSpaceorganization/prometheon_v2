# The subnet DB layer `/v2` contract

The interface the subnet DB layer implements and this repository consumes. It is
the only bridge between the BitFan platform and the subnet.

The authoritative definition is the code, not this page:

- **Schemas**: [`src/prometheon/dbclient/models.py`](../src/prometheon/dbclient/models.py)
- **Authentication**: [`src/prometheon/dbclient/auth.py`](../src/prometheon/dbclient/auth.py)
- **A working implementation**: [`src/prometheon/dbclient/fake.py`](../src/prometheon/dbclient/fake.py)

`FakeDbLayer` is a complete in-memory implementation of everything below. Build
against it first; it enforces the same rules a real server must.

---

## 1. Authentication

**There are no API keys.** A caller proves identity by signing a request
envelope with its hotkey, and the server checks that signature against the
metagraph. A deregistered hotkey loses access the moment the chain says so, with
nobody having to remember to revoke anything.

Six headers:

```http
X-Prometheon-Netuid:      481
X-Prometheon-Hotkey:      5F…
X-Prometheon-Nonce:       <32 lowercase hex>
X-Prometheon-Issued-At:   <unix seconds>
X-Prometheon-Body-Sha256: <64 lowercase hex>
X-Prometheon-Signature:   0x<128 hex>
```

The signed payload is canonicalised (JCS: sorted keys, no whitespace, no
floats), prefixed with the domain string and a newline, and signed SR25519.

### 1.1 The signed bytes, exactly

This is the normative example. Every literal below is pinned by
`tests/contract/test_db_api_contract.py`; if your implementation produces
different bytes for these inputs, it will not interoperate.

```text
PROMETHEON_V2_DB_REQUEST\n
{"body_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
"domain":"PROMETHEON_V2_DB_REQUEST",
"hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
"issued_at":1786032000,
"method":"GET",
"netuid":42,
"nonce":"9f1c0d3a7b5e42610fd8c2b41a6e9037",
"path":"/v2/dataset/2026-08-05/test?cursor=eyJjIjoidGVzdCIsImQiOiIyMDI2LTA4LTA1IiwibyI6NTAwfQ&limit=500"}
```

Written across lines here for legibility. **On the wire the JSON carries no
whitespace at all.** The newline appears only once, between the domain string
and the opening brace.

Three things to note:

- `domain` appears **both** as the prefix and as a field. Belt and braces: the
  prefix makes a signature from one domain unusable in another even if a verifier
  forgets to check the field.
- `path` is the full origin-form target, **query string included**, exactly as it
  appears on the request line.
- `body_sha256` of an empty body is the SHA-256 of the empty string,
  `e3b0c442…7852b855`. Every GET carries it.

The other two domains are `PROMETHEON_V2_EVALUATION` and
`PROMETHEON_V2_MODEL_COMMITMENT`.

Every field closes a specific hole, and a server implementer who drops one will
not see anything break:

- **`method` and `path`** bind the signature to one route. Without them a signed
  read of yesterday's manifest is a signed read of anything.
- **`body_sha256`** binds the payload. Without it a captured POST envelope
  authenticates a substituted body.
- **`netuid`** stops an envelope signed for testnet authenticating on mainnet.
- **`nonce` and `issued_at`** bound replay to 300 seconds and then to nothing.

### Four rules a server must not get wrong

**Rebuild `method`, `path` and `body_sha256` from what you received.** Never from
the headers. The headers exist so the client can say what it signed; if they
disagree with the request, the signature does not verify. Verifying against
client-supplied values makes all three fields decorative.

**`path` is the origin-form request target**, query string included, exactly as
it appears on the wire. Neither side normalises.

**Claim the nonce *after* the signature verifies.** Claiming first lets anyone
who can observe a nonce burn it by replaying an unsigned request. That turns a
replay defence into a denial-of-service tool against the caller it protects.

**Expire nonces against your own clock.** `issued_at` is supplied by the caller
and is therefore not a clock. Pruning against it lets a replayer choose an
`issued_at` at the edge of the skew window, advance your horizon, and evict the
record that would have caught them. Retention must exceed twice the skew window:
900s against 300s.

Reject a malformed header with `401 db.auth_malformed`, including one that is
only too long. It must never become a 500.

---

## 2. Endpoints

| Method | Path | Caller | Returns |
|---|---|---|---|
| GET | `/v2/snapshot/{date}` | validator | `SnapshotManifest` |
| GET | `/v2/dataset/{date}/test` | validator, miner¹ | `TestContentPage` |
| GET | `/v2/dataset/{date}/production` | validator | `ProductionContentPage` |
| GET | `/v2/eligible-miners/{date}` | validator | `EligibleMinerSet` |
| GET | `/v2/models/{date}` | validator | `ModelCommitmentSet` |
| POST | `/v2/evaluations` | validator | `EvaluationAccepted` |

¹ Miners may read test content only after the embargo.

`{date}` is `YYYY-MM-DD`, UTC.

Constants: `SNAPSHOT_VERSION = "prometheon-snapshot/2"`, `EMBARGO_DAYS = 2`,
`PRODUCTION_ITEM_CAP = 10000`, `DEFAULT_PAGE_LIMIT = 500`,
`MAX_PAGE_LIMIT = 1000`, `MAX_CONTENT_CHARS = 32000`.

---

## 3. Pagination

`?cursor=<base64url>&limit=<n>`. A response carries `next_cursor`, or `null` on
the last page.

**The cursor must strictly advance.** A client aborts if it is handed a cursor it
has already been served at any point, not just the one it sent a moment ago. A
server alternating two cursors would otherwise re-yield the same items until the
page cap, which looks like a large dataset rather than a fault.

---

## 4. Content rules

**Test content is attributed; production content is not.** `TestContentItem`
carries `author_hotkey`; `ProductionContentItem` carries **no author field of any
kind**, and clients reject an unknown field rather than ignoring it. Production
content is where the negatives come from, so attribution there would be a direct
lever on scoring.

**Truncate at ingest.** Content longer than `MAX_CONTENT_CHARS` must be truncated
before it is stored, never at read time and never by the client. If truncation
happened later, the labeller would judge full text while models read a truncated
version. That is an irreducible corpus error no model could overcome.

**The embargo is two days.** Day N's content is readable by miners at
**00:00 UTC on day N+2**, by which point the 04:00 cycle that used it has long
finished. Serve `403 db.embargoed` before that, and carry `available_at` so the
caller does not have to guess when to return. Validators are never embargoed.

Midnight, not 04:00. The boundary is a property of the *date*, so every server
and every client derives it from the date alone with no reference to when any
particular validator happens to run. `FakeDbLayer` lifts it at midnight and
`test_it_lifts_at_midnight_two_days_later` pins that to the second.

The embargo covers **production content too**, not only test content. The table
above lists `/v2/dataset/{date}/production` as validator-only, and a server that
role-gated it would answer `db.not_authorized` where a miner is owed
`db.embargoed` — a different code with a different remediation.

---

## 5. The snapshot manifest

```json
{
  "date": "2026-08-05",
  "snapshot_version": "prometheon-snapshot/2",
  "built_at": 1754400000,
  "test_item_count": 1840,
  "production_item_count": 8000,
  "eligible_miner_count": 11,
  "model_commitment_count": 9,
  "content_hash": "<64 hex>"
}
```

`content_hash` is computed by `snapshot_content_hash()` over every collection.
Every validator recomputes it from what it was served and refuses the day if it
disagrees.

**This is the check that keeps the DB layer honest.** A server that served two
different corpora to two validators would be caught by both of them at ingest,
rather than surfacing weeks later as unexplained weight divergence. Implement it
by hashing the same way the client does, or the mismatch is yours, not theirs.

### 5.1 Worked example

Every value here is pinned by `tests/contract/test_db_api_contract.py`. If your
implementation produces different digests for these inputs, it does not conform.

```python
date              = 2026-08-05
test_items        = [TestContentItem(id="t1", content="example",
                                     author_hotkey="5Grwva…GKutQY",
                                     submitted_at=1786000000)]
production_items  = [ProductionContentItem(id="p1", content="hello",
                                           observed_at=1786000001)]
eligible_miners   = [EligibleMiner(hotkey="5Grwva…GKutQY", qualified_user_count=12)]
model_commitments = [ModelCommitment(hotkey="5Grwva…GKutQY", hf_repo="miner/model",
                                     revision_sha="aaaa…aaaa", chute_id="chute-1",
                                     committed_at=1785999000, block=4242)]
```

| Value | Digest |
|---|---|
| `test_item_digest(t1)` | `a1a6993cd3e2b94d2a92e577769bd8f6e2cc80c7d909fc1d52e2fc57167be5fb` |
| `collection_hash([t1])` | `afb2b787d6f4f92acf239a7d0cd2a53037ae1df1f5c500b20fc63a31b6510748` |
| `collection_hash([])` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `snapshot_content_hash(…)` | `093c1302b549a9c06c543295ec35e5a3b06c91626831effbf17a53954d9bdab3` |
| an entirely empty day | `f1afe099299e07dabf31b211b6dcee506d5d5fe95333978154c74c5becb77f92` |

An empty collection hashes to the digest of the empty string, so a day with no
content of some kind still produces a well-defined snapshot hash rather than a
special case.

Return `409 db.snapshot_not_ready` until the day is complete and the snapshot is
built. A snapshot can only be complete once the day is over.

`ModelCommitmentSet.frozen_at` must be **00:00 UTC of `date`**. Validators assert
it. It exists so a commitment written at 03:59 cannot change what one validator
scores and another does not.

---

## 6. Errors

Always this shape, never a bare status:

```json
{"error": {"code": "db.embargoed", "message": "content for 2026-08-06 is released at 2026-08-08T00:00:00Z", "available_at": 1786233600}}
```

| Code | Status | Meaning |
|---|---|---|
| `db.bad_request` | 400 | malformed date or query |
| `db.bad_cursor` | 400 | cursor not base64url, or not one you issued |
| `db.auth_malformed` | 401 | a header is missing or the wrong shape |
| `db.auth_invalid` | 401 | signature does not verify |
| `db.auth_expired` | 401 | `issued_at` outside ±300s |
| `db.auth_replayed` | 401 | nonce already spent |
| `db.not_authorized` | 403 | hotkey has no role for this route |
| `db.embargoed` | 403 | correct caller, too early |
| `db.not_found` | 404 | no such date |
| `db.snapshot_not_ready` | 409 | the day is not built yet |
| `db.duplicate_evaluation` | 409 | this validator already submitted this date |
| `db.invalid_payload` | 422 | body failed schema validation |
| `db.rate_limited` | 429 | back off; honour `Retry-After` |

Clients key off `code`, not status, because status is too coarse to act on: 403
is both "you are not a validator" and "come back in two days", and those have
different remediations.

`Retry-After` must be **delta-seconds** (digits only) and greater than zero. A
`Retry-After: 0` is treated as absent. Honouring it literally would turn the one
status that means *back off* into a tight retry loop.

---

## 7. Evaluations

`POST /v2/evaluations` takes an `EvaluationSubmission` signed under
`prometheon.evaluation.v1`. Verify the signature against `validator_hotkey`, and
reject a second submission for the same `(date, validator_hotkey)` with
`409 db.duplicate_evaluation`.

**Every number in it is an integer.** Ratios travel as basis points or as
fixed-point micros, named so in the field, because the payload is canonicalised
for signing and canonical JSON refuses floats.

Store them. They are what makes the subnet auditable: anyone holding the same
day's snapshot can recompute a validator's numbers and check its signature,
without trusting the DB layer that served them.

---

## 8. Building against it

```python
from prometheon.dbclient import DbClient, FakeDbLayer

layer = FakeDbLayer(netuid=481)
layer.register(hotkey, CallerRole.VALIDATOR)
layer.seed_day(day, test_items=…, production_items=…)

with DbClient(base_url="https://db.invalid", netuid=481,
              keypair=keypair, transport=layer.transport) as db:
    snapshot = db.fetch_day(day)
```

`FakeDbLayer.request_log` records `(method, request target)` for everything that
arrived, authenticated or not, so a test can assert what actually went on the
wire rather than what a mock agreed to.
