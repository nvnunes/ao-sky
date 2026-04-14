"""Thin command-line entrypoints for ao-sky."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._version import __version__
from .about import describe_package


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="ao-sky",
        description="AO-sky mapping and Gaia-backed guide-star tooling.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.set_defaults(handler=_handle_status)
    subparsers = parser.add_subparsers(dest="command")

    status_parser = subparsers.add_parser(
        "status",
        help="Show the current bootstrap-stage package summary.",
    )
    status_parser.set_defaults(handler=_handle_status)
    return parser


def _handle_status(_: argparse.Namespace) -> int:
    print(describe_package())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
