"""Behaviour of the wire types and of the snapshot content hash.

The hash tests are the important ones. Everything a validator does downstream
assumes that two validators handed the same day derive the same hash, and that
any difference in what they were handed shows up as a different hash. Both
halves of that need proving.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError
from tests.unit.dbclient.factories import (
    DAY,
    make_commitment,
    make_production_items,
    make_test_items,
)

from prometheon.dbclient.models import (
    EligibleMiner,
    EvaluationSubmission,
    MinerResult,
    ModelCommitment,
    ModelStatus,
    ProductionContentItem,
    SnapshotManifest,
    corpus_content_hash,
    snapshot_content_hash,
)
from prometheon.dbclient.models import TestContentItem as SubmittedTestItem
from prometheon.security.canonical_json import to_canonical_bytes

pytestmark = pytest.mark.unit

HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
OTHER_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def _hash(
    *,
    day: dt.date = DAY,
    test_items: tuple[SubmittedTestItem, ...] | None = None,
    production_items: tuple[ProductionContentItem, ...] | None = None,
    eligible: tuple[EligibleMiner, ...] = (),
    commitments: tuple[ModelCommitment, ...] = (),
) -> str:
    return snapshot_content_hash(
        date=day,
        test_items=test_items if test_items is not None else make_test_items(3, author=HOTKEY),
        production_items=(
            production_items if production_items is not None else make_production_items(3)
        ),
        eligible_miners=eligible,
        model_commitments=commitments,
    )


class TestSnapshotContentHash:
    def test_is_independent_of_the_order_items_arrive_in(self) -> None:
        items = make_test_items(5, author=HOTKEY)
        assert _hash(test_items=items) == _hash(test_items=tuple(reversed(items)))

    def test_changing_one_character_of_content_changes_it(self) -> None:
        items = make_test_items(3, author=HOTKEY)
        edited = (*items[:-1], items[-1].model_copy(update={"content": "test content 3 "}))
        assert _hash(test_items=items) != _hash(test_items=edited)

    def test_reattributing_a_test_item_changes_it(self) -> None:
        """Attribution is scored, so it has to be inside the hash."""
        items = make_test_items(3, author=HOTKEY)
        moved = (*items[:-1], items[-1].model_copy(update={"author_hotkey": OTHER_HOTKEY}))
        assert _hash(test_items=items) != _hash(test_items=moved)

    def test_dropping_an_item_changes_it(self) -> None:
        items = make_test_items(4, author=HOTKEY)
        assert _hash(test_items=items) != _hash(test_items=items[:-1])

    def test_duplicating_an_item_changes_it(self) -> None:
        """A set-shaped fold would miss this; multiplicity has to survive."""
        items = make_test_items(3, author=HOTKEY)
        assert _hash(test_items=items) != _hash(test_items=(*items, items[0]))

    def test_the_same_data_on_a_different_day_hashes_differently(self) -> None:
        assert _hash(day=DAY) != _hash(day=DAY + dt.timedelta(days=1))

    def test_moving_an_item_between_collections_changes_it(self) -> None:
        """Positives and negatives must not be interchangeable in the hash."""
        production = make_production_items(3)
        as_test = tuple(
            SubmittedTestItem(
                id=item.id,
                content=item.content,
                author_hotkey=HOTKEY,
                claimed_violating=True,
                submitted_at=item.observed_at,
            )
            for item in production
        )
        assert _hash(test_items=(), production_items=production) != _hash(
            test_items=as_test, production_items=()
        )

    def test_eligible_miners_and_commitments_are_covered(self) -> None:
        baseline = _hash()
        with_miner = _hash(eligible=(EligibleMiner(hotkey=HOTKEY, qualified_user_count=3),))
        with_commitment = _hash(commitments=(make_commitment(HOTKEY),))
        assert len({baseline, with_miner, with_commitment}) == 3

    def test_qualified_user_count_is_covered(self) -> None:
        low = _hash(eligible=(EligibleMiner(hotkey=HOTKEY, qualified_user_count=3),))
        high = _hash(eligible=(EligibleMiner(hotkey=HOTKEY, qualified_user_count=4),))
        assert low != high

    def test_an_empty_day_still_hashes(self) -> None:
        digest = _hash(test_items=(), production_items=())
        assert len(digest) == 64
        assert digest == _hash(test_items=(), production_items=())


class TestCorpusContentHash:
    def test_is_independent_of_labelling_order(self) -> None:
        labels = [("a", True), ("b", False), ("c", True)]
        assert corpus_content_hash(date=DAY, labelled=labels) == corpus_content_hash(
            date=DAY, labelled=list(reversed(labels))
        )

    def test_a_flipped_label_changes_it(self) -> None:
        assert corpus_content_hash(date=DAY, labelled=[("a", True)]) != corpus_content_hash(
            date=DAY, labelled=[("a", False)]
        )


class TestContentModels:
    def test_production_content_rejects_an_author_field(self) -> None:
        """The client is the enforcement point for de-attributed production data."""
        with pytest.raises(ValidationError):
            ProductionContentItem.model_validate(
                {"id": "p1", "content": "x", "observed_at": 1, "author_hotkey": HOTKEY}
            )

    def test_a_moving_revision_reference_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_commitment(HOTKEY, revision="main")

    def test_an_uppercase_revision_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_commitment(HOTKEY, revision="A" * 40)

    def test_a_malformed_hotkey_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EligibleMiner(hotkey="not-an-address", qualified_user_count=1)

    def test_a_manifest_content_hash_must_be_hex(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotManifest.model_validate(
                {
                    "date": DAY.isoformat(),
                    "snapshot_version": "prometheon-snapshot/2",
                    "built_at": 0,
                    "test_item_count": 0,
                    "production_item_count": 0,
                    "eligible_miner_count": 0,
                    "model_commitment_count": 0,
                    "content_hash": "Z" * 64,
                }
            )

    def test_production_count_is_capped(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotManifest.model_validate(
                {
                    "date": DAY.isoformat(),
                    "snapshot_version": "prometheon-snapshot/2",
                    "built_at": 0,
                    "test_item_count": 0,
                    "production_item_count": 10_001,
                    "eligible_miner_count": 0,
                    "model_commitment_count": 0,
                    "content_hash": "0" * 64,
                }
            )


def _submission(**overrides: object) -> EvaluationSubmission:
    base: dict[str, object] = {
        "date": DAY,
        "netuid": 42,
        "validator_hotkey": HOTKEY,
        "scoring_version": "prometheon-scoring/2.0.0",
        "policy_version": "2026-08-07",
        "snapshot_content_hash": "a" * 64,
        "corpus_content_hash": "b" * 64,
        "labelled_test_count": 4,
        "labelled_production_count": 7,
        "results": (
            MinerResult(
                hotkey=HOTKEY,
                uid=3,
                weight=1234,
                dataset_submitted_count=10,
                dataset_valid_count=8,
                dataset_score_micro=6_400_000,
                model_status=ModelStatus.EVALUATED,
                model_items_scored=100,
                model_correct_count=91,
                raw_accuracy_bp=9100,
                mean_total_tokens_milli=1_250_000,
                efficiency_penalty_bp=900,
                model_score_bp=8281,
                model_rank=1,
            ),
        ),
        "burned_weight": 0,
        "started_at": 1,
        "completed_at": 2,
    }
    base.update(overrides)
    return EvaluationSubmission.model_validate(base)


class TestEvaluationSubmission:
    def test_the_signing_payload_carries_no_floats(self) -> None:
        """Canonical encoding is the gate; this proves the shape gets through it."""
        assert to_canonical_bytes(_submission().signing_payload())

    def test_the_signature_is_not_part_of_what_it_signs(self) -> None:
        unsigned = _submission()
        signed = _submission(signature="0x" + "ab" * 64)
        assert unsigned.signing_payload() == signed.signing_payload()

    def test_every_scored_field_is_covered_by_the_signature(self) -> None:
        baseline = _submission().signing_payload()
        tampered = _submission(
            results=(_submission().results[0].model_copy(update={"weight": 9999}),)
        ).signing_payload()
        assert baseline != tampered
