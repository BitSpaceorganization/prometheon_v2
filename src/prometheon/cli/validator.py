"""Validator commands: run a cycle, and pull a released dataset.

``run`` wires the real world into :mod:`prometheon.cli.cycle` and does the two
things a cycle cannot do for itself: submit weights, and publish the record.
Both are gated by config so a new operator can watch a full cycle run without
touching the chain.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from prometheon.chain import subtensor as chain
from prometheon.cli._common import (
    EXIT_OK,
    load,
    note,
    open_wallet,
    out,
    require_env,
    table,
)
from prometheon.cli.cycle import (
    SCORING_VERSION,
    CycleResult,
    build_results,
    run_cycle,
    submit_allocation,
    submit_weights,
)
from prometheon.config import Config, ScoreSource
from prometheon.dbclient import auth
from prometheon.dbclient.client import DbClient
from prometheon.dbclient.models import EvaluationSubmission
from prometheon.errors import ConfigError
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.validation import ModelRegistry
from prometheon.scoring.mirrored import verify_mirrored

#: A cycle scores *yesterday*: today's content is still arriving.
_DEFAULT_LAG_DAYS = 1


def _target_day(args: argparse.Namespace) -> dt.date:
    if args.date:
        try:
            return dt.date.fromisoformat(args.date)
        except ValueError as exc:
            raise ConfigError(f"--date must be YYYY-MM-DD, got {args.date!r}") from exc
    return dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=_DEFAULT_LAG_DAYS)


#: The policy is Markdown, so its version line is written for a reader:
#: ``**Version:** 2026-08-07``. Matching it must therefore tolerate emphasis
#: markers. A plain ``startswith("version:")`` did not, so the *shipped* policy
#: file failed this check and `validator run` refused to start on a correct
#: repository. The parser was wrong, not the document.
_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*[*_]{0,2}version[*_]{0,2}\s*:\s*[*_]{0,2}\s*(?P<version>[^*_\s][^*_]*?)\s*[*_]{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _policy_version(text: str) -> str:
    """The declared version of a policy document, or ``""``."""
    match = _VERSION_RE.search(text)
    return match.group("version").strip() if match else ""


def _policy_text(args: argparse.Namespace) -> tuple[str, str]:
    """The policy every verdict is judged against, and its version line.

    Read from a file rather than embedded, so a validator can prove which text
    it used and two validators can diff theirs. The version goes into the signed
    record, which makes a mid-cycle policy change visible instead of merely
    confusing.
    """
    path = Path(args.policy) if args.policy else Path("content_policy.md")
    if not path.is_file():
        raise ConfigError(
            f"no content policy at {path}. Pass --policy, or run from the "
            "repository root where content_policy.md lives"
        )
    text = path.read_text(encoding="utf-8")
    version = _policy_version(text)
    if not version:
        raise ConfigError(
            f"{path} carries no 'Version:' line. Every verdict is published "
            "against a policy version so a disagreement can be attributed"
        )
    return text, version


#: Where a completed cycle leaves the vector it submitted, so it can be posted
#: again between cycles.
_LAST_WEIGHTS_FILE: Final[str] = "last-weights.json"


def _state_path(config: Config) -> Path:
    return config.validator.state_directory / _LAST_WEIGHTS_FILE


def _write_allocation(
    config: Config, *, day: dt.date, weights: Mapping[str, int], burn_hotkey: str
) -> Path:
    """Write the state file `validator resubmit` reads.

    One writer for both modes: a mirrored vector has to be re-posted between
    cycles exactly like a computed one, and a second copy of this would be a
    second place for the format to drift.
    """
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "day": day.isoformat(),
                "netuid": config.chain.netuid,
                "burn_hotkey": burn_hotkey,
                "scoring_version": SCORING_VERSION,
                "submitted_at": int(time.time()),
                "weights": dict(weights),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def save_submitted_allocation(config: Config, *, day: dt.date, result: CycleResult) -> Path:
    """Record the allocation a cycle submitted.

    A cycle runs once a day, but weights stop counting toward consensus once
    ``activity_cutoff`` passes -- 720 blocks, under three hours, on a subnet
    whose tempo is 72 minutes. Between cycles the validator has to re-post the
    same vector or its miners earn nothing for the rest of the day, so the
    vector has to outlive the process that computed it.
    """
    return _write_allocation(
        config, day=day, weights=result.split.weights, burn_hotkey=result.burn_hotkey
    )


def cmd_resubmit(args: argparse.Namespace) -> int:
    """Re-post the last cycle's vector without recomputing it.

    Deliberately cheap: no labelling, no evaluation, no GPU time, no spend.
    It reads what the last cycle decided and sends it again, resolved against a
    *fresh* metagraph by the same code path the cycle itself uses -- so a miner
    that deregistered since is dropped here too, rather than being paid because
    a re-post was allowed to skip the rule.
    """
    config = load(args.config)
    path = _state_path(config)
    if not path.is_file():
        raise ConfigError(
            f"no submitted allocation at {path}; run `prometheon validator run` first, "
            "and note that a dry run does not leave one"
        )

    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state.get("netuid", -1)) != config.chain.netuid:
        raise ConfigError(
            f"{path} was written for netuid={state.get('netuid')}, "
            f"but this config targets netuid={config.chain.netuid}"
        )
    weights = {str(key): int(value) for key, value in dict(state["weights"]).items()}
    if not weights:
        raise ConfigError(f"{path} carries no weights to submit")

    if config.validator.dry_run:
        note(f"dry run: would re-submit {len(weights)} hotkeys from {state['day']}")
        return EXIT_OK

    wallet = open_wallet(config)
    subtensor = chain.connect(config.chain.network)
    capabilities = chain.detect_capabilities(subtensor)

    # A cycle and a re-post landing inside the rate limit is routine -- both run
    # on timers, and the cycle's own submission is the one that matters. Exiting
    # non-zero here would mark a healthy validator as failing once a day.
    waiting = chain.blocks_until_weights_allowed(
        subtensor, netuid=config.chain.netuid, hotkey=wallet.hotkey.ss58_address
    )
    if waiting:
        note(f"rate limit not clear for another {waiting} blocks; nothing sent")
        return EXIT_OK

    receipt = submit_allocation(
        weights=weights,
        burn_hotkey=str(state["burn_hotkey"]),
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        mechid=0 if capabilities.supports_mechid else None,
    )
    note(f"re-submitted {state['day']} ({len(weights)} hotkeys): {receipt or 'no receipt'}")
    return EXIT_OK


def cmd_mirror(args: argparse.Namespace) -> int:
    """Submit another validator's published scores instead of computing any.

    Reads the record ``score_provider`` published for the day, verifies it, and
    sends its weights. No labelling, no model evaluation, no GPU -- and no
    independent opinion, which is the trade `ScoringConfig.score_source`
    describes.
    """
    config = load(args.config)
    day = _target_day(args)
    provider = config.scoring.score_provider
    wallet = open_wallet(config)
    note(
        f"validator {wallet.hotkey.ss58_address} mirroring {provider} "
        f"for {day.isoformat()} on netuid={config.chain.netuid}"
    )

    subtensor = chain.connect(config.chain.network)
    metagraph = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
    burn_hotkey = chain.read_subnet_owner_hotkey(subtensor, netuid=config.chain.netuid)

    with DbClient(
        base_url=config.db.base_url,
        netuid=config.chain.netuid,
        keypair=wallet.hotkey,
        timeout_seconds=config.db.request_timeout_seconds,
        max_retries=config.db.max_retries,
    ) as db:
        note(f"fetching {provider}'s record for {day.isoformat()}")
        submission = db.get_evaluation(day, provider)
        # The snapshot this validator can see for itself, so a record describing
        # a different corpus is caught rather than trusted.
        snapshot = db.get_snapshot(day)
        weights = verify_mirrored(
            submission,
            provider=provider,
            day=day,
            netuid=config.chain.netuid,
            metagraph=metagraph,
            burn_hotkey=burn_hotkey,
            snapshot_content_hash=snapshot.content_hash,
        )

    out(
        table(
            [
                ("day", day.isoformat()),
                ("provider", provider),
                ("scoring version", submission.scoring_version),
                ("snapshot", submission.snapshot_content_hash[:12]),
                ("corpus", submission.corpus_content_hash[:12]),
                ("weighted hotkeys", str(len(weights))),
                ("burn", str(submission.burned_weight)),
            ]
        )
    )

    if config.validator.dry_run:
        note("")
        note("dry run: the record verified but nothing was submitted")
        return EXIT_OK
    if not config.validator.submit_weights:
        note("submit_weights is off: the record verified but nothing was sent")
        return EXIT_OK

    capabilities = chain.detect_capabilities(subtensor)
    receipt = submit_allocation(
        weights=weights,
        burn_hotkey=burn_hotkey,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        mechid=0 if capabilities.supports_mechid else None,
    )
    note(f"weights submitted: {receipt or 'no receipt returned'}")
    saved = _write_allocation(config, day=day, weights=weights, burn_hotkey=burn_hotkey)
    note(f"allocation saved for re-submission: {saved}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Run one cycle: label, evaluate, score, submit."""
    config = load(args.config)
    if config.scoring.score_source is ScoreSource.ENDPOINT:
        # `validator run` stays the one command an operator schedules, whichever
        # mode they chose; the config decides what it does.
        return cmd_mirror(args)
    day = _target_day(args)
    policy, policy_version = _policy_text(args)

    # Read here purely to fail fast with a message that says what the key is
    # for. The labelling client reads it again from the same variable when it is
    # built; checking twice is cheaper than discovering it four hours in.
    require_env(config.labelling.api_key_env, purpose="label the day's corpus as ground truth")

    wallet = open_wallet(config)
    hotkey = wallet.hotkey.ss58_address
    note(f"validator {hotkey} scoring {day.isoformat()} on netuid={config.chain.netuid}")

    subtensor = chain.connect(config.chain.network)
    started = int(time.time())

    with (
        DbClient(
            base_url=config.db.base_url,
            netuid=config.chain.netuid,
            keypair=wallet.hotkey,
            timeout_seconds=config.db.request_timeout_seconds,
            max_retries=config.db.max_retries,
        ) as db,
        HuggingFaceClient() as huggingface,
        OpenAICompatibleClient(config.labelling) as labeller,
    ):
        result = run_cycle(
            config=config,
            day=day,
            db=db,
            subtensor=subtensor,
            registry=ModelRegistry(
                huggingface=huggingface,
                max_weight_bytes=config.evaluation.max_weight_bytes,
            ),
            labeller=labeller,
            policy=policy,
            policy_version=policy_version,
            now=started,
            progress=note,
        )

        rows = build_results(result)
        completed = int(time.time())

        out(_summary(result, rows))
        for message in result.notes:
            note(f"note: {message}")

        if config.validator.dry_run:
            note("")
            note("dry run: no weights submitted, no results published")
            return EXIT_OK

        if config.validator.submit_weights:
            capabilities = chain.detect_capabilities(subtensor)
            receipt = submit_weights(
                result,
                config=config,
                subtensor=subtensor,
                wallet=wallet,
                mechid=0 if capabilities.supports_mechid else None,
            )
            note(f"weights submitted: {receipt or 'no receipt returned'}")
            saved = save_submitted_allocation(config, day=day, result=result)
            note(f"allocation saved for re-submission: {saved}")
        else:
            note("submit_weights is off: weights were computed but not sent")

        if config.validator.submit_results:
            submission = auth.sign_evaluation(
                wallet.hotkey,
                EvaluationSubmission(
                    date=day,
                    netuid=config.chain.netuid,
                    validator_hotkey=hotkey,
                    scoring_version=SCORING_VERSION,
                    policy_version=policy_version,
                    snapshot_content_hash=result.snapshot.manifest.content_hash,
                    corpus_content_hash=result.corpus_hash,
                    labelled_test_count=sum(
                        1 for item in result.corpus.items if item.source.value == "test"
                    ),
                    labelled_production_count=sum(
                        1 for item in result.corpus.items if item.source.value == "production"
                    ),
                    results=rows,
                    burned_weight=result.split.burn_units,
                    started_at=started,
                    completed_at=completed,
                ),
            )
            accepted = db.submit_evaluation(submission)
            note(f"results published: {accepted.model_dump_json()}")
        else:
            note("submit_results is off: this cycle is not auditable by anyone else")

    return EXIT_OK


def cmd_schedule(args: argparse.Namespace) -> int:
    """Run the cycle daily and the re-post every half hour, in this process.

    The alternative to two crontab entries, for a host that has no cron -- and
    the two jobs share a process, so a cycle that runs long cannot have a
    re-post submit weights underneath it. No lock file needed.

    Wake-ups are aligned to the wall clock, so every validator running this
    re-posts on the same :00/:30 instants however long its own cycle took.
    """
    from prometheon.cli.schedule import run_forever

    config = load(args.config)
    mode = config.scoring.score_source.value
    note(
        f"scheduler started: cycle at {args.cycle_hour:02d}:00 UTC in {mode} mode, "
        f"re-post on :00 and :30"
    )

    def cycle() -> None:
        # Fresh Namespace per run: `--date` must stay unset so each cycle
        # targets its own yesterday rather than the day the process started.
        cmd_run(argparse.Namespace(config=args.config, date=None))

    def resubmit() -> None:
        cmd_resubmit(argparse.Namespace(config=args.config, date=None))

    run_forever(
        cycle=cycle,
        resubmit=resubmit,
        cycle_hour=args.cycle_hour,
        say=note,
        max_iterations=args.max_iterations,
    )
    return EXIT_OK


def cmd_dataset_pull(args: argparse.Namespace) -> int:
    """Download a released day's test content as JSONL.

    Subject to the embargo: a day's test content is released two days later, so
    the corpus a cycle scored cannot be studied until after it was used.
    """
    config = load(args.config)
    day = _target_day(args)
    wallet = open_wallet(config)

    with DbClient(
        base_url=config.db.base_url,
        netuid=config.chain.netuid,
        keypair=wallet.hotkey,
        timeout_seconds=config.db.request_timeout_seconds,
        max_retries=config.db.max_retries,
    ) as db:
        items = db.fetch_test_content(day)

    lines = [
        json.dumps(
            {"id": item.id, "content": item.content, "author_hotkey": item.author_hotkey},
            ensure_ascii=False,
            sort_keys=True,
        )
        for item in items
    ]
    if args.output:
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        note(f"wrote {len(lines)} items to {args.output}")
    else:
        for line in lines:
            out(line)
    return EXIT_OK


def _summary(result: object, rows: tuple[object, ...]) -> str:
    from prometheon.cli.cycle import CycleResult

    assert isinstance(result, CycleResult)
    eligible = [item for item in result.validations if item.valid]
    return table(
        [
            ("day", result.day.isoformat()),
            ("snapshot", result.snapshot.manifest.content_hash),
            ("corpus", result.corpus_hash),
            ("corpus items", str(len(result.corpus.items))),
            (
                "labelled",
                f"{len(result.labelled.labels)} ok, {len(result.labelled.excluded_ids)} excluded",
            ),
            ("labelling calls", str(result.labelled.calls)),
            ("models eligible", f"{len(eligible)}/{len(result.validations)}"),
            ("models evaluated", str(len(result.evaluations))),
            ("efficiency lambda", f"{result.ranking.applied_lambda_bp}bp"),
            ("token spread", f"{result.ranking.mean_token_cv_micros / 10_000:.2f}%"),
            ("weighted hotkeys", str(len(result.split.weights))),
            ("burn", f"{result.split.burn_units} units to {result.burn_hotkey}"),
            ("published rows", str(len(rows))),
        ]
    )


__all__ = ["cmd_dataset_pull", "cmd_run"]
