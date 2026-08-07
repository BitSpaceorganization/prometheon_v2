"""The ``prometheon`` command-line surface.

Thin. Every command is a few lines of wiring around a library function, so
anything worth testing is tested where it lives rather than through a
subprocess, and an operator's workflow and a script's import path exercise the
same code.
"""

from __future__ import annotations

from prometheon.cli.cycle import SCORING_VERSION, CycleResult, build_results, run_cycle
from prometheon.cli.main import build_parser, main

__all__ = [
    "SCORING_VERSION",
    "CycleResult",
    "build_parser",
    "build_results",
    "main",
    "run_cycle",
]
