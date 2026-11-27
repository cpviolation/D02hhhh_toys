"""Command-line entry point: ``d02hhhh-toys setup OUTPUT``, ``d02hhhh-toys study CONFIG --out DIR`` and ``d02hhhh-toys summarise DIR``."""

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
    study.add_argument(
        "--store-toys",
        action="store_true",
        help="write each toy to OUT/toys/ (about 34 MB per toy at 1e5 events per tag)",
    )
    study.add_argument(
        "--toys-from",
        metavar="DIR",
        help="read the toys from DIR/toys/ (a study with the same generation settings)",
    )
    setup = commands.add_parser(
        "setup", help="first-time setup: link results/ to an output folder of choice"
    )
    setup.add_argument("output", help="folder for the outputs (created if missing)")
    setup.add_argument("--link", default="results", help="the link (default: results)")
    setup.add_argument(
        "--move", action="store_true", help="move an existing results/ into OUTPUT"
    )
    setup.add_argument(
        "--force", action="store_true", help="replace a link to another folder"
    )
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
            config,
            args.out,
            workers=args.workers,
            store_toys=args.store_toys,
            toys_from=args.toys_from,
            log=lambda m: print(m, flush=True),
        )
        return 0
    if args.command == "setup":
        from d02hhhh_toys.workspace import SetupError, link_output

        try:
            target = link_output(
                args.output, args.link, move=args.move, force=args.force
            )
        except SetupError as error:
            print(f"setup: {error}", file=sys.stderr)
            return 1
        print(f"{args.link} -> {target}")
        return 0
    if args.command == "summarise":
        from d02hhhh_toys.study import summarise

        summarise(args.out)
        return 0
    parser.print_help(sys.stderr)
    return 1
