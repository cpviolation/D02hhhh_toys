"""Command-line entry point (the study driver is added in Phase 7)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="d02hhhh-toys", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print the version")
    args = parser.parse_args(argv)
    if args.version:
        from d02hhhh_toys.version import version

        print(version)
    return 0
