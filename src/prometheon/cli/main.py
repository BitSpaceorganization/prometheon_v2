"""The ``prometheon`` command.

Argument parsing and error rendering only. Every command body lives in a sibling
module so it can be tested without building a parser.

Two conventions hold throughout:

**Results on stdout, progress on stderr.** ``prometheon dataset pull`` can be
piped straight into a file while still printing what it is doing to a terminal.

**No traceback ever reaches an operator.** Every typed failure renders as one
greppable ``error [code] message`` line with a non-zero exit. A traceback means
this package missed a case, so it is a defect here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from prometheon.cli import miner, validator
from prometheon.cli._common import EXIT_FAILED, EXIT_USAGE, fail, note
from prometheon.errors import PrometheonError
from prometheon.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prometheon",
        description="Prometheon V2 — an open moderation-model competition on Bittensor.",
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

    render = with_config(
        model.add_parser(
            "render",
            help="render the canonical wrapper to deploy on Chutes",
            description=(
                "Print the exact miner.py to deploy with `chutes deploy`. It carries "
                "the inference engine as well as the chute definition, and is checked "
                "against the canonical hash before it is written, so a script no "
                "validator would accept is caught before deployment rather than after "
                "a day of serving. It also prints the chute id to commit, which is "
                "derived from your account and hotkey and so is knowable before the "
                "chute exists."
            ),
        )
    )
    render.add_argument("--hf-repo")
    render.add_argument("--hf-revision", help="40-character commit SHA, not a branch")
    render.add_argument("--chutes-user")
    render.add_argument(
        "--gpu-count", type=int, default=1, help="GPUs per instance (1-8); you pay for them"
    )
    render.add_argument("--min-vram-gb", type=int, default=16, help="VRAM floor per GPU (16-140)")
    render.add_argument("--output", type=Path, help="write here instead of stdout")
    render.set_defaults(handler=miner.cmd_render)

    commit = with_config(
        model.add_parser(
            "commit",
            help="write (hf_repo, hf_revision, chute_id) on chain",
            description=(
                "This is the submission. Note that re-committing re-dates your claim "
                "on a revision, so do not re-commit an unchanged model without reason."
            ),
        )
    )
    commit.add_argument("--hf-repo")
    commit.add_argument("--hf-revision", help="40-character commit SHA, not a branch")
    commit.add_argument("--chute-id")
    commit.add_argument(
        "--dry-run", action="store_true", help="print the payload without writing it"
    )
    commit.set_defaults(handler=miner.cmd_commit)

    verify = with_config(
        model.add_parser(
            "verify",
            help="run the validator's own eligibility checks against your deployment",
            description=(
                "Runs the same ModelRegistry a validator runs, against the commitment "
                "currently on chain. Exits non-zero if your model would not be scored."
            ),
        )
    )
    verify.add_argument("--hf-repo")
    verify.add_argument("--hf-revision", help="check these values instead of the chain")
    verify.add_argument("--chute-id")
    verify.add_argument("--chutes-api-key", help="raises the Chutes rate limit; optional")
    verify.set_defaults(handler=miner.cmd_verify)

    # -- canonical --------------------------------------------------------
    canonical = commands.add_parser(
        "canonical", help="the hashes every deployment is verified against"
    )
    canonical.set_defaults(handler=miner.cmd_canonical)

    # -- dataset ----------------------------------------------------------
    dataset = commands.add_parser("dataset", help="released evaluation data").add_subparsers(
        dest="subcommand", required=True
    )
    pull = with_config(
        dataset.add_parser(
            "pull",
            help="download a released day's test content as JSONL",
            description=(
                "Subject to the two-day embargo: a day's test content is released "
                "after the evaluation that used it has finished."
            ),
        )
    )
    pull.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    pull.add_argument("--output", type=Path, help="write here instead of stdout")
    pull.set_defaults(handler=validator.cmd_dataset_pull)

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


#: Closing a pipe early is what `| head` does; it is not a failure.
EXIT_OK_ON_PIPE = 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
