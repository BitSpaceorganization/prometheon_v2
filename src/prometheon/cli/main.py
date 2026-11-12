"""The ``prometheon`` command.

Argument parsing and error rendering only. Every command body lives in a sibling
module so it can be tested without building a parser.

Two conventions hold throughout:

**Results on stdout, progress on stderr.** A command's summary can be piped
straight into a file while it still prints what it is doing to a terminal.

**No traceback ever reaches an operator.** Every typed failure renders as one
greppable ``error [code] message`` line with a non-zero exit. A traceback means
this package missed a case, so it is a defect here.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from prometheon.cli import credits, miner, update, validator
from prometheon.cli._common import EXIT_FAILED, EXIT_USAGE, fail, note
from prometheon.errors import PrometheonError
from prometheon.version import SUBMISSION_FEE_ALPHA, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prometheon",
        description=(
            "Prometheon, a Bittensor subnet building an AI moderation agent, "
            "starting with text moderation."
        ),
    )
    parser.add_argument("--version", action="version", version=f"prometheon {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def with_config(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument(
            "--config",
            required=True,
            help="path to the TOML config file",
        )
        return sub

    # -- model ------------------------------------------------------------
    model = commands.add_parser(
        "model", help="publish, commit and check a moderation model"
    ).add_subparsers(dest="subcommand", required=True)

    commit = with_config(
        model.add_parser(
            "commit",
            help="write (hf_repo, hf_revision) on chain",
            description=(
                "This is the submission. The hub freezes commitments once a day, and "
                "a model with new weights needs one submission credit (see "
                "'prometheon credits'). Re-committing the same weights costs nothing "
                "and keeps the block they were first committed at."
            ),
        )
    )
    commit.add_argument("--hf-repo")
    commit.add_argument("--hf-revision", help="40-character commit SHA, not a branch")
    commit.add_argument(
        "--dry-run", action="store_true", help="print the payload without writing it"
    )
    commit.add_argument(
        "--if-changed",
        action="store_true",
        help=(
            "write only if the chain does not already hold this repo and revision. "
            "Use this for anything on a timer: an unchanged re-commit changes nothing "
            "but still pays a transaction fee"
        ),
    )
    commit.set_defaults(handler=miner.cmd_commit)

    verify = with_config(
        model.add_parser(
            "verify",
            help="run the validator's own eligibility checks against your model",
            description=(
                "Runs the same ModelRegistry a validator runs, against the commitment "
                "currently on chain. Exits non-zero if your model would not be scored."
            ),
        )
    )
    verify.add_argument("--hf-repo")
    verify.add_argument("--hf-revision", help="check these values instead of the chain")
    verify.set_defaults(handler=miner.cmd_verify)

    # -- credits ----------------------------------------------------------
    credits_commands = commands.add_parser(
        "credits", help="submission credits: read a balance, buy credits"
    ).add_subparsers(dest="subcommand", required=True)
    balance = with_config(
        credits_commands.add_parser(
            "balance",
            help="show a coldkey's credits and ledger, from the hub's public route",
        )
    )
    balance.add_argument("--address", help="coldkey or hotkey (default: your wallet's coldkey)")
    balance.add_argument("--limit", type=int, default=10, help="ledger entries to show")
    balance.set_defaults(handler=credits.cmd_balance)

    buy = with_config(
        credits_commands.add_parser(
            "buy",
            help="pay for credits: transfer_stake from your coldkey to the treasury",
            description=(
                "Moves alpha stake on the subnet from your coldkey to the treasury coldkey, "
                f"one credit per {SUBMISSION_FEE_ALPHA} alpha. Prints the transfer first. "
                "There are no refunds."
            ),
        )
    )
    buy.add_argument("--credits", type=int, default=1, help="credits to buy (default: 1)")
    buy.add_argument("--hotkey", help="the hotkey your stake is on (default: your wallet's)")
    buy.add_argument("--dry-run", action="store_true", help="print the transfer, send nothing")
    buy.add_argument("--yes", action="store_true", help="send without asking to confirm")
    buy.set_defaults(handler=credits.cmd_buy)

    # -- validator --------------------------------------------------------
    validator_commands = commands.add_parser(
        "validator", help="run the daily cycle"
    ).add_subparsers(dest="subcommand", required=True)
    run = with_config(
        validator_commands.add_parser(
            "run",
            help="label, evaluate, score and submit one day",
            description=(
                "Runs one full cycle. Set [validator] dry_run = true to compute "
                "everything without touching the chain or publishing a result."
            ),
        )
    )
    run.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    run.add_argument("--policy", help="path to content_policy.md")
    run.add_argument(
        "--force",
        action="store_true",
        help=(
            "compute the day again even if this validator already computed it; by default "
            "a run resumes from the last step it completed"
        ),
    )
    run.set_defaults(handler=validator.cmd_run)

    resubmit = with_config(
        validator_commands.add_parser(
            "resubmit",
            help="re-post the last cycle's weights without recomputing them",
            description=(
                "Sends the vector the last `validator run` submitted, resolved against "
                "a fresh metagraph. Weights stop counting toward consensus once "
                "activity_cutoff passes, which is far shorter than a day, so a "
                "once-daily cycle must re-post between runs or its miners earn nothing "
                "for the rest of the day. Costs nothing: no labelling, no evaluation."
            ),
        )
    )
    resubmit.set_defaults(handler=validator.cmd_resubmit)

    schedule = with_config(
        validator_commands.add_parser(
            "schedule",
            help="run the cycle daily and the re-post every 30 minutes, in this process",
            description=(
                "The alternative to cron or systemd. The cycle runs on its own thread, "
                "so re-posts keep the last vector counting while it works. Wake-ups "
                "align to the wall clock, so every validator running this re-posts on "
                "the same :00 and :30 instants."
            ),
        )
    )
    schedule.add_argument(
        "--cycle-hour",
        type=int,
        default=4,
        choices=range(24),
        metavar="HOUR",
        help="UTC hour to start the daily cycle (default: 4)",
    )
    schedule.add_argument("--policy", help="path to content_policy.md")
    schedule.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    schedule.set_defaults(handler=validator.cmd_schedule)

    # -- update -----------------------------------------------------------
    update_commands = commands.add_parser(
        "update", help="move this install to a newer release tag, or back"
    ).add_subparsers(dest="subcommand", required=True)
    with_config(
        update_commands.add_parser(
            "status", help="the running release, the current and previous ones, what is newer"
        )
    ).set_defaults(handler=update.cmd_status)
    with_config(
        update_commands.add_parser(
            "check", help="fetch release tags from origin and say whether one is newer"
        )
    ).set_defaults(handler=update.cmd_check)
    apply = with_config(
        update_commands.add_parser(
            "apply",
            help="install the newest release tag beside this one, check it, switch to it",
            description=(
                "Only release tags vX.Y.Z from this install's own origin, newer than the "
                "running version: never a branch, never an older version. The new release "
                "is installed beside this one and checks itself before anything switches; "
                "if any step fails, this release stays current. Waits for a running cycle."
            ),
        )
    )
    apply.add_argument(
        "--scheduled",
        action="store_true",
        help=(
            "as the timer runs it: do nothing unless [update] auto is on, and skip rather "
            "than wait while a cycle runs"
        ),
    )
    apply.set_defaults(handler=update.cmd_apply)
    with_config(
        update_commands.add_parser("rollback", help="switch back to the previous release")
    ).set_defaults(handler=update.cmd_rollback)
    with_config(
        # Run by `update apply`, from inside the release it is about to switch to.
        update_commands.add_parser("self-check", help=argparse.SUPPRESS)
    ).set_defaults(handler=update.cmd_self_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # `exc.code or EXIT_USAGE` looks harmless and turns 0 into 2, because
        # `0 or 2` is 2. argparse exits 0 for --version and --help, so the two
        # commands a new user runs first both reported failure, breaking
        # `set -e` wrappers and container healthchecks.
        code = exc.code
        return EXIT_USAGE if code is None else int(code)

    try:
        result = args.handler(args)
        return int(result)
    except PrometheonError as exc:
        return fail(exc)
    except KeyboardInterrupt:
        note("interrupted")
        return EXIT_FAILED
    except BrokenPipeError:  # pragma: no cover - `| head` closing the pipe
        return EXIT_OK_ON_PIPE
    except Exception as exc:
        # A case this package missed. One greppable line rather than a
        # traceback, with the way to get the traceback when it is needed (B24).
        if os.environ.get("PROMETHEON_DEBUG"):
            raise
        note(
            f"error [internal] {type(exc).__name__}: {exc}. This is a bug in prometheon; "
            "rerun with PROMETHEON_DEBUG=1 for the traceback and report it"
        )
        return EXIT_FAILED


#: Closing a pipe early is what `| head` does; it is not a failure.
EXIT_OK_ON_PIPE = 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
