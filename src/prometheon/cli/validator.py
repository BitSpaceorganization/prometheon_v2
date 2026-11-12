"""Validator commands: run a cycle, re-post its weights, mirror, schedule.

``run`` wires the real world into :mod:`prometheon.cli.cycle` and does the two
things a cycle cannot do for itself: submit weights, and publish the record.
Both are gated by config so a new operator can watch a full cycle run without
touching the chain.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

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
    Policy,
    build_results,
    run_cycle,
    submit_allocation,
)
from prometheon.cli.record import draft_record, finish_record
from prometheon.cli.state import CycleState, load_state, save_state, state_path
from prometheon.config import Config, ScoreSource
from prometheon.errors import ConfigError, HubError, PrometheonError
from prometheon.hub import auth
from prometheon.hub.client import HubClient
from prometheon.hub.models import EvaluationSubmission
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


def _policy_text(args: argparse.Namespace) -> Policy:
    """The policy every verdict is judged against, its version and its hash.

    Read from a file rather than embedded, so a validator can prove which text
    it used. The hash is compared with the one the day's epoch froze, so a
    validator labelling against a different text stops instead of computing
    different ground truth. The version goes into the signed record.
    """
    path = Path(getattr(args, "policy", None) or "content_policy.md")
    if not path.is_file():
        raise ConfigError(
            f"no content policy at {path}. Pass --policy, or run from the "
            "repository root where content_policy.md lives"
        )
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path} is not UTF-8 text: {exc}") from exc
    version = _policy_version(text)
    if not version:
        raise ConfigError(
            f"{path} carries no 'Version:' line. Every verdict is published "
            "against a policy version so a disagreement can be attributed"
        )
    if len(version) > 64:
        raise ConfigError(f"{path} declares a version over 64 characters: {version[:70]!r}")
    return Policy(text=text, version=version, sha256=hashlib.sha256(raw).hexdigest())


def _hub_client(config: Config, *, keypair: Any = None) -> HubClient:
    """A client for the configured hub, signing as ``keypair`` when given one."""
    return HubClient(
        base_url=config.hub.base_url,
        netuid=config.chain.netuid,
        epoch_keys=config.hub.epoch_keys,
        keypair=keypair,
        timeout_seconds=config.hub.request_timeout_seconds,
        max_retries=config.hub.max_retries,
    )


#: A re-post of a vector this many days old says so loudly: the cycles that
#: should have replaced it are failing, and miners are being paid on old work.
_STALE_REPOST_DAYS: Final[int] = 2


#: One send at a time from this process. Under `validator schedule` the cycle
#: runs on its own thread while re-posts continue on the main one.
_SEND_LOCK = threading.Lock()


def _send(config: Config, state: CycleState, *, wallet: Any, subtensor: Any) -> CycleState:
    """Send the state's vector and record what was actually sent."""
    capabilities = chain.detect_capabilities(subtensor)
    with _SEND_LOCK:
        sent = submit_allocation(
            weights=state.weights,
            burn_hotkey=state.burn_hotkey,
            config=config,
            subtensor=subtensor,
            wallet=wallet,
            mechid=0 if capabilities.supports_mechid else None,
        )
    state.sent_at = int(time.time())
    state.sent_weights = dict(sent.units)
    state.sent_uids = dict(sent.uids)
    state.gone = list(sent.gone)
    state.receipt = sent.receipt
    save_state(config, state)
    if sent.gone:
        note(f"{len(sent.gone)} hotkey(s) had deregistered; their units went to burn")
    if sent.padded:
        note(
            f"the subnet requires more weights than this vector had: {len(sent.padded)} uid(s) "
            "were added with the smallest possible weight"
        )
    if sent.capped:
        note(
            f"{len(sent.capped)} share(s) were cut to the subnet's max_weight_limit; the excess "
            "was burned, or shared among the other entries where the burn was over the limit"
        )
    note(f"weights submitted for {state.day.isoformat()}: {sent.receipt or 'no receipt returned'}")
    return state


def _publish(config: Config, state: CycleState, *, wallet: Any, hub: HubClient) -> None:
    """Sign the day's record once, save it, then post those exact bytes.

    Raises :class:`HubError` when the hub cannot be reached; the signed record
    stays saved, and the next ``validator run`` or ``validator resubmit`` posts
    the same bytes again, which the hub accepts as a duplicate if the first
    attempt had in fact landed.
    """
    if state.record is None or state.published_at is not None:
        return
    if not state.signed:
        draft = EvaluationSubmission.model_validate(state.record)
        # After the send, so the record describes the vector as it was sent:
        # units of hotkeys that deregistered in between counted as burn (B5),
        # and any shaping to the chain's weight limits (S5).
        finished = finish_record(
            draft,
            sent=state.sent_weights if state.sent_weights is not None else state.weights,
            burn_hotkey=state.burn_hotkey,
            uids=state.sent_uids or {},
        )
        signed = auth.sign_evaluation(wallet.hotkey, finished)
        state.record = json.loads(signed.model_dump_json())
        state.signed = True
        save_state(config, state)
    record = EvaluationSubmission.model_validate(state.record)
    try:
        accepted = hub.submit_result(record)
    except HubError as exc:
        if exc.status_code == 409:
            # A different record for this day is already stored, from a run
            # whose state was lost. The hub keeps the first; so does this.
            state.published_at = int(time.time())
            save_state(config, state)
            note(f"the hub already holds another record for {state.day.isoformat()}; kept that one")
            return
        raise
    state.published_at = int(time.time())
    save_state(config, state)
    note(f"results published{' (already stored)' if accepted.duplicate else ''}")


#: How many days back a mirror will look for a record newer than the one it
#: holds. A mirror that fell behind did so because its daily run failed, and
#: three days is well past the point where an operator should have noticed.
#: It also bounds the request count: a mirror that is already current sends
#: none at all.
MIRROR_LOOKBACK_DAYS: Final[int] = 3


def _mirrored_for(
    hub: Any,
    config: Config,
    *,
    day: dt.date,
    provider: str,
    metagraph: Any,
    burn_hotkey: str,
) -> Mapping[str, int] | None:
    """One day's verified weights, or ``None`` if this is not a day to use.

    ``None`` covers the two ordinary cases together, because the caller treats
    them identically: the provider has not published this day yet, and the
    record it did publish does not verify against the pinned hotkey. Neither is
    worth a log line on a timer that fires every thirty minutes; both mean the
    caller should look at an older day, and failing that keep what it holds.
    """
    try:
        submission = hub.get_result(day, provider)
        verified = hub.get_public_epoch(day)
        return verify_mirrored(
            submission,
            provider=provider,
            day=day,
            netuid=config.chain.netuid,
            metagraph=metagraph,
            burn_hotkey=burn_hotkey,
            epoch_content_hash=verified.content_hash,
        )
    except Exception:
        return None


def _refresh_mirrored(
    config: Config,
    *,
    stored_day: dt.date,
    wallet: Any,
    subtensor: Any,
    burn_hotkey: str,
    today: dt.date,
) -> tuple[dt.date, Mapping[str, int]] | None:
    """The newest record the provider has published after ``stored_day``.

    ``None`` means keep what is on disk, and that is the ordinary outcome.

    A mirror fetched its provider's record once, when `validator run` last
    fired, and re-posted that vector every half hour until the next daily run.
    If that run happened before the provider published -- which is exactly what
    a provider retrying a failed cycle causes -- the mirror's run failed, wrote
    nothing, and it spent the next day re-posting the *previous* day's vector.
    Observed on mainnet under v2 on 2026-09-11: a provider published three and a half
    hours late after a retry, and 22.5% of validating stake sat on the day
    before for the rest of the day.

    Re-reading the day already held would not have helped, because a published
    record is immutable: the hub accepts an identical resend and refuses a
    revision. The only thing worth looking for is a *newer* day, which is what
    this does.

    A mirror that is current asks for nothing: there are no candidate days, so
    the whole function is a date comparison. Only one that is behind spends a
    request, and then at most ``MIRROR_LOOKBACK_DAYS`` of them.

    **Never raises.** Weights stop counting once ``activity_cutoff`` passes, so
    a hub that is down, or a day the provider has not published yet, must
    not stop this validator submitting. Both are ordinary and neither is worth
    a line in the log; the caller falls back to the vector on disk.
    """
    provider = config.scoring.score_provider
    # A cycle scores the day before it runs, so nothing newer than yesterday
    # can exist. Newest first: the freshest record wins and the rest are
    # never requested.
    candidates: list[dt.date] = []
    day = today - dt.timedelta(days=1)
    while day > stored_day and len(candidates) < MIRROR_LOOKBACK_DAYS:
        candidates.append(day)
        day -= dt.timedelta(days=1)
    if not candidates:
        return None

    try:
        metagraph = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
        # Public reads: a mirror needs no signature to fetch a published record.
        with _hub_client(config) as hub:
            for candidate in candidates:
                weights = _mirrored_for(
                    hub,
                    config,
                    day=candidate,
                    provider=provider,
                    metagraph=metagraph,
                    burn_hotkey=burn_hotkey,
                )
                if weights is not None:
                    return candidate, weights
    except Exception:
        # Could not even reach the hub. Keep re-posting; see the docstring.
        return None
    return None


@chain.closes_connections
def cmd_resubmit(args: argparse.Namespace) -> int:
    """Re-post the last cycle's vector, and post its record if still pending.

    Deliberately cheap: no labelling, no evaluation, no GPU time, no spend.
    It reads what the last cycle decided and sends it again, resolved against a
    *fresh* metagraph by the same code path the cycle itself uses, so a miner
    that deregistered since goes to burn here too.
    """
    config = load(args.config)
    state = load_state(config)
    if state is None:
        raise ConfigError(
            f"no saved vector at {state_path(config)}; run `prometheon validator run` first, "
            "and note that a dry run does not leave one"
        )
    if state.netuid != config.chain.netuid:
        raise ConfigError(
            f"{state_path(config)} was written for netuid={state.netuid}, "
            f"but this config targets netuid={config.chain.netuid}"
        )
    if not state.weights:
        raise ConfigError(f"{state_path(config)} carries no weights to submit")

    if config.validator.dry_run:
        note(f"dry run: would re-submit {len(state.weights)} hotkeys from {state.day.isoformat()}")
        return EXIT_OK
    if not config.validator.submit_weights:
        # v2 kept re-posting with this off, so an operator who switched it off
        # to investigate still sent the old vector every hour (B15).
        note("submit_weights is off: nothing re-posted")
        return EXIT_OK

    age = (dt.datetime.now(dt.timezone.utc).date() - state.day).days
    if age > _STALE_REPOST_DAYS:
        note(
            f"warning: re-posting the vector for {state.day.isoformat()}, {age} days old. The "
            "daily cycle has not replaced it; check its log"
        )

    wallet = open_wallet(config)
    subtensor = chain.connect(config.chain.network)

    # A cycle and a re-post landing inside the rate limit is routine -- both run
    # on timers, and the cycle's own submission is the one that matters. Exiting
    # non-zero here would mark a healthy validator as failing once a day.
    waiting = chain.blocks_until_weights_allowed(
        subtensor, netuid=config.chain.netuid, hotkey=wallet.hotkey.ss58_address
    )
    if waiting:
        note(f"rate limit not clear for another {waiting} blocks; nothing sent")
    else:
        # In endpoint mode every re-post looks for a newer provider record
        # than the one held. A mirror that only replays cannot catch up.
        if config.scoring.score_source is ScoreSource.ENDPOINT:
            found = _refresh_mirrored(
                config,
                stored_day=state.day,
                wallet=wallet,
                subtensor=subtensor,
                burn_hotkey=state.burn_hotkey,
                today=dt.datetime.now(dt.timezone.utc).date(),
            )
            if found is not None:
                day, refreshed = found
                note(
                    f"{config.scoring.score_provider} has published {day.isoformat()}, newer "
                    f"than the {state.day.isoformat()} vector held"
                )
                state = CycleState(
                    day=day,
                    netuid=config.chain.netuid,
                    burn_hotkey=state.burn_hotkey,
                    scoring_version=SCORING_VERSION,
                    weights=dict(refreshed),
                    computed_at=int(time.time()),
                )
                save_state(config, state)
        _send(config, state, wallet=wallet, subtensor=subtensor)

    if config.validator.submit_results and state.record is not None and state.published_at is None:
        try:
            with _hub_client(config, keypair=wallet.hotkey) as hub:
                _publish(config, state, wallet=wallet, hub=hub)
        except PrometheonError as exc:
            note(f"the record for {state.day.isoformat()} is still waiting to be published: {exc}")
    return EXIT_OK


def _wait_for_weights_window(
    subtensor: Any,
    *,
    config: Config,
    wallet: Any,
    attempts: int = 40,
    sleep: Any = time.sleep,
) -> None:
    """Block until this hotkey may set weights again.

    Bounded: `weights_set_rate_limit` is a subnet hyperparameter, commonly 100
    blocks (~20 min), so the window always clears. The cap exists so a chain that never reports the
    limit clearing fails loudly instead of hanging a scheduler forever.
    """
    for _ in range(attempts):
        waiting = chain.blocks_until_weights_allowed(
            subtensor, netuid=config.chain.netuid, hotkey=wallet.hotkey.ss58_address
        )
        if waiting <= 0:
            return
        note(f"rate limit: {waiting} blocks (~{waiting * 12 / 60:.0f} min) before weights land")
        sleep(min(waiting, 10) * 12 + 6)
    raise ConfigError(
        "the weights rate limit never cleared; refusing to spin. Something else "
        "is submitting with this hotkey, or the chain is not advancing"
    )


@chain.closes_connections
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

    with _hub_client(config) as hub:
        note(f"fetching {provider}'s record for {day.isoformat()}")
        submission = hub.get_result(day, provider)
        # The signed epoch this validator can verify for itself, so a record
        # describing a different day is caught rather than trusted.
        verified = hub.get_public_epoch(day)
        weights = verify_mirrored(
            submission,
            provider=provider,
            day=day,
            netuid=config.chain.netuid,
            metagraph=metagraph,
            burn_hotkey=burn_hotkey,
            epoch_content_hash=verified.content_hash,
        )

    out(
        table(
            [
                ("day", day.isoformat()),
                ("provider", provider),
                ("scoring version", submission.scoring_version),
                ("epoch", submission.epoch_content_hash[:12]),
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

    # Wait the rate limit out rather than raising into it. A mirror cycle and a
    # re-post both run on timers and land inside the 100-block (~20 min) window
    # routinely; `cmd_resubmit` skips because it retries in thirty minutes, but
    # this call has a specific day's vector to land and skipping would lose it.
    #
    # Raising is what the previous version did, and under a supervisor that is
    # a restart loop: the process exits non-zero, comes back, re-runs the whole
    # cycle, and hits the same limit. Observed three times over before one
    # submission landed.
    state = CycleState(
        day=day,
        netuid=config.chain.netuid,
        burn_hotkey=burn_hotkey,
        scoring_version=SCORING_VERSION,
        weights=dict(weights),
        computed_at=int(time.time()),
    )
    save_state(config, state)
    _wait_for_weights_window(subtensor, config=config, wallet=wallet)
    _send(config, state, wallet=wallet, subtensor=subtensor)
    return EXIT_OK


@chain.closes_connections
def cmd_run(args: argparse.Namespace) -> int:
    """Run one cycle: label, evaluate, score, then send and publish.

    Each step is saved before the next one starts, so a run interrupted after
    the cycle computed resumes from the step it reached instead of labelling
    and evaluating the day again. ``--force`` computes it again regardless.
    """
    config = load(args.config)
    if config.scoring.score_source is ScoreSource.ENDPOINT:
        # `validator run` stays the one command an operator schedules, whichever
        # mode they chose; the config decides what it does.
        return cmd_mirror(args)
    day = _target_day(args)
    policy = _policy_text(args)
    if not config.hub.epoch_keys:
        raise ConfigError(
            "no hub key is pinned under [hub.epoch_keys], so the day's epoch cannot be "
            "verified. Copy the key from the release notes of the version you run"
        )

    state = load_state(config)
    resuming = (
        state is not None
        and state.day == day
        and state.netuid == config.chain.netuid
        and state.scoring_version == SCORING_VERSION
        and not getattr(args, "force", False)
    )
    if resuming and state is not None and state.is_complete(config):
        note(f"{day.isoformat()} is already scored, sent and published; nothing to do")
        return EXIT_OK

    wallet = open_wallet(config)
    hotkey = wallet.hotkey.ss58_address
    subtensor = chain.connect(config.chain.network)

    with _hub_client(config, keypair=wallet.hotkey) as hub:
        if resuming and state is not None:
            note(
                f"{day.isoformat()} was computed already; resuming without labelling or "
                "evaluating it again (pass --force to recompute)"
            )
        else:
            # Read here purely to fail fast with a message that says what the
            # key is for, before an hour of work rather than after it.
            require_env(
                config.labelling.api_key_env, purpose="label the day's corpus as ground truth"
            )
            note(f"validator {hotkey} scoring {day.isoformat()} on netuid={config.chain.netuid}")
            started = int(time.time())
            with (
                HuggingFaceClient() as huggingface,
                OpenAICompatibleClient(config.labelling) as labeller,
            ):
                result = run_cycle(
                    config=config,
                    day=day,
                    hub=hub,
                    subtensor=subtensor,
                    registry=ModelRegistry(
                        huggingface=huggingface,
                        max_weight_bytes=config.evaluation.max_weight_bytes,
                        max_non_weight_bytes=config.evaluation.max_non_weight_bytes,
                    ),
                    labeller=labeller,
                    policy=policy,
                    now=started,
                    progress=note,
                )
            completed = int(time.time())
            out(_summary(result, build_results(result)))
            for message in result.notes:
                note(f"note: {message}")

            if config.validator.dry_run:
                note("")
                note("dry run: no weights submitted, no results published")
                return EXIT_OK

            draft = draft_record(
                result,
                netuid=config.chain.netuid,
                validator_hotkey=hotkey,
                policy_version=policy.version,
                started_at=started,
                completed_at=completed,
            )
            state = CycleState(
                day=day,
                netuid=config.chain.netuid,
                burn_hotkey=result.burn_hotkey,
                scoring_version=SCORING_VERSION,
                weights=dict(result.split.weights),
                computed_at=completed,
                record=json.loads(draft.model_dump_json()),
            )
            # Saved before anything is sent. v2 saved only after a successful
            # send, so a send refused inside the rate limit lost the day (B3).
            note(f"vector saved before sending: {save_state(config, state)}")

        assert state is not None
        if config.validator.dry_run:
            note("dry run: nothing sent or published")
            return EXIT_OK

        if not config.validator.submit_weights:
            note("submit_weights is off: weights were computed but not sent")
        elif state.sent_at is None:
            # Wait the rate limit out rather than failing into it: a re-post
            # that landed minutes earlier is routine, not an error (B3).
            _wait_for_weights_window(subtensor, config=config, wallet=wallet)
            state = _send(config, state, wallet=wallet, subtensor=subtensor)

        if not config.validator.submit_results:
            note("submit_results is off: this cycle is not auditable by anyone else")
        else:
            _publish(config, state, wallet=wallet, hub=hub)

    return EXIT_OK


def _last_completed_cycle_day(config: Config) -> dt.date | None:
    """The day the last cycle finished: sent and published as configured.

    ``None`` when there is no state, it cannot be read, or the day it holds is
    not finished, all of which mean the scheduler should run (or resume) a
    cycle. The cost of resuming one is small now; the cost of wrongly skipping
    one is a day nobody is paid for.
    """
    try:
        state = load_state(config)
    except ConfigError:
        return None
    if state is None or not state.is_complete(config):
        return None
    return state.day


def cmd_schedule(args: argparse.Namespace) -> int:
    """Run the cycle daily and re-post every half hour, in this process.

    The alternative to cron or systemd. The cycle runs on its own thread, so
    re-posts keep the last vector counting while a local cycle works for hours.
    Wake-ups are aligned to the wall clock, so every validator running this
    re-posts on the same :00/:30 instants.
    """
    from prometheon.cli.schedule import run_forever

    config = load(args.config)
    mode = config.scoring.score_source.value
    note(
        f"scheduler started: cycle at {args.cycle_hour:02d}:00 UTC in {mode} mode, "
        f"re-post on :00 and :30"
    )

    def cycle(day: dt.date) -> None:
        # v2 built this Namespace without `policy`, so in local mode every
        # scheduled cycle died on an AttributeError and nothing was scored (B1).
        cmd_run(
            argparse.Namespace(
                config=args.config,
                date=day.isoformat(),
                policy=getattr(args, "policy", None),
                force=False,
            )
        )

    def resubmit() -> None:
        try:
            state = load_state(config)
        except ConfigError as exc:
            note(f"re-post skipped: {exc}")
            return
        if state is None:
            note("nothing to re-post yet: no cycle has computed a vector")
            return
        cmd_resubmit(argparse.Namespace(config=args.config, date=None))

    already = _last_completed_cycle_day(config)
    if already is not None:
        note(f"last finished cycle on record: {already.isoformat()}")

    run_forever(
        cycle=cycle,
        resubmit=resubmit,
        cycle_hour=args.cycle_hour,
        say=note,
        last_finished_day=lambda: _last_completed_cycle_day(config),
        max_iterations=args.max_iterations,
    )
    return EXIT_OK


def _summary(result: object, rows: tuple[object, ...]) -> str:
    from prometheon.cli.cycle import CycleResult

    assert isinstance(result, CycleResult)
    eligible = [item for item in result.validations if item.valid]
    feed = result.data.feed
    return table(
        [
            ("day", result.day.isoformat()),
            ("epoch", result.epoch_content_hash),
            ("frozen block", str(result.epoch.frozen_block)),
            ("corpus", result.corpus_hash),
            (
                "feed lines",
                f"{len(feed.test)} test, {len(feed.production)} production, "
                f"{len(feed.rejected)} rejected",
            ),
            ("eligible miners", f"{len(result.eligible)}/{len(result.epoch.miners)}"),
            ("corpus items", str(len(result.corpus.items))),
            (
                "labelled",
                f"{len(result.labelled.labels)} ok, {len(result.labelled.excluded_ids)} excluded",
            ),
            ("labelling calls", str(result.labelled.calls)),
            ("models verified", f"{len(eligible)}/{len(result.validations)}"),
            ("models evaluated", str(len(result.evaluations))),
            ("efficiency lambda", f"{result.ranking.applied_lambda_bp}bp"),
            ("token spread", f"{result.ranking.mean_token_cv_micros / 10_000:.2f}%"),
            ("weighted hotkeys", str(len(result.split.weights))),
            ("burn", f"{result.split.burn_units} units to {result.burn_hotkey}"),
            ("published rows", str(len(rows))),
        ]
    )


__all__ = ["cmd_mirror", "cmd_resubmit", "cmd_run", "cmd_schedule"]
