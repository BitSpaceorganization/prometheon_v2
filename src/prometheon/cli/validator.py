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
from pathlib import Path
from typing import Final

import httpx

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
from prometheon.cli.cycle import SCORING_VERSION, build_results, run_cycle, submit_weights
from prometheon.dbclient import auth
from prometheon.dbclient.client import DbClient
from prometheon.dbclient.models import EvaluationSubmission
from prometheon.errors import ConfigError
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.registry.chutes import ChutesClient
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.validation import ModelRegistry

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


def cmd_run(args: argparse.Namespace) -> int:
    """Run one cycle: label, evaluate, score, submit."""
    config = load(args.config)
    day = _target_day(args)
    policy, policy_version = _policy_text(args)

    # Read here purely to fail fast with a message that says what the key is
    # for. The labelling client reads it again from the same variable when it is
    # built; checking twice is cheaper than discovering it four hours in.
    require_env(config.labelling.api_key_env, purpose="label the day's corpus as ground truth")
    chutes_key = require_env(
        config.evaluation.chutes_api_key_env, purpose="call miners' deployed models"
    )

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
        ChutesClient(api_key=chutes_key) as chutes,
        OpenAICompatibleClient(config.labelling) as labeller,
        httpx.Client(
            timeout=httpx.Timeout(config.evaluation.request_timeout_seconds)
        ) as evaluation_client,
    ):
        result = run_cycle(
            config=config,
            day=day,
            db=db,
            subtensor=subtensor,
            registry=ModelRegistry(huggingface=huggingface, chutes=chutes),
            labeller=labeller,
            evaluation_client=evaluation_client,
            policy=policy,
            policy_version=policy_version,
            chutes_api_key=chutes_key,
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
