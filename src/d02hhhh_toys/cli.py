"""Command-line entry point: ``d02hhhh-toys study CONFIG --out DIR`` and ``d02hhhh-toys summarise DIR``."""

from __future__ import annotations

import argparse
import os
import sys


def _single_threaded() -> None:
    """One BLAS/OpenMP thread per process: the parallelism is over toys."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="d02hhhh-toys", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print the version")
    commands = parser.add_subparsers(dest="command")
    study = commands.add_parser("study", help="run or resume a toy study")
    study.add_argument("config", help="study YAML (configs/study/*.yaml)")
    study.add_argument("--out", required=True, help="output directory (results/...)")
    study.add_argument("--workers", type=int, default=1, help="parallel toy processes")
    study.add_argument("--toys", type=int, help="override the number of toys")
    summary = commands.add_parser("summarise", help="rewrite summary.json and plots")
    summary.add_argument("out", help="study directory")
    args = parser.parse_args(argv)
    if args.version:
        from d02hhhh_toys.version import version

        print(version)
        return 0
    if args.command == "study":
        if args.workers > 1:
            _single_threaded()
        from d02hhhh_toys.study import StudyConfig, run_study

        config = StudyConfig.from_yaml(args.config)
        if args.toys is not None:
            config.n_toys = args.toys
        run_study(
            config, args.out, workers=args.workers, log=lambda m: print(m, flush=True)
        )
        return 0
    if args.command == "summarise":
        from d02hhhh_toys.study import summarise

        summarise(args.out)
        return 0
    parser.print_help(sys.stderr)
    return 1
