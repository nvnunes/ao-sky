"""Thin command-line entrypoints for ao-sky."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
        help="Show the current package summary.",
    )
    status_parser.set_defaults(handler=_handle_status)

    init_parser = subparsers.add_parser(
        "init",
        help="Create a new persisted build root from a build-definition YAML file.",
    )
    init_parser.add_argument("definition", type=Path, help="Build-definition YAML filename.")
    init_parser.add_argument("--gaia-root", type=Path, default=None, help="Resolved Gaia root.")
    init_parser.add_argument("--build-root", type=Path, default=None, help="Resolved build root.")
    init_parser.add_argument("--dust-root", type=Path, default=None, help="Resolved dust root.")
    init_parser.add_argument(
        "--aosky-conf",
        type=Path,
        default=None,
        help="Optional aosky.conf YAML with gaia_root/build_root/dust_root defaults.",
    )
    init_parser.add_argument(
        "--legacy-config",
        type=Path,
        default=None,
        help="Temporary legacy config.yaml used to load AO-system runtime policy.",
    )
    init_parser.set_defaults(handler=_handle_init)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one initialized build to completion.",
    )
    run_parser.add_argument("build", type=Path, help="Build directory to run.")
    run_parser.set_defaults(handler=_handle_run)

    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart the latest build in one AO-system/config lineage.",
    )
    restart_parser.add_argument("ao_system_short_name", help="AO-system lineage name.")
    restart_parser.add_argument("config_short_name", help="Config lineage name.")
    restart_parser.add_argument("--build-root", type=Path, default=None, help="Resolved build root.")
    restart_parser.add_argument(
        "--aosky-conf",
        type=Path,
        default=None,
        help="Optional aosky.conf YAML with build_root defaults.",
    )
    restart_parser.set_defaults(handler=_handle_restart)

    show_parser = subparsers.add_parser(
        "show",
        help="Show a persisted build summary.",
    )
    show_parser.add_argument("build", type=Path, help="Build directory to summarize.")
    show_parser.set_defaults(handler=_handle_show)
    return parser


def _handle_status(_: argparse.Namespace) -> int:
    print(describe_package())
    return 0


def _handle_init(args: argparse.Namespace) -> int:
    from .build import init_build
    from .build.config import DEFAULT_LEGACY_CONFIG

    build_path = init_build(
        definition_filename=args.definition,
        gaia_root=args.gaia_root,
        build_root=args.build_root,
        dust_root=args.dust_root,
        aosky_conf=args.aosky_conf,
        legacy_config_path=args.legacy_config or DEFAULT_LEGACY_CONFIG,
    )
    print(build_path)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    from .build import run_build

    print(run_build(args.build))
    return 0


def _handle_restart(args: argparse.Namespace) -> int:
    from .build import restart_build

    print(
        restart_build(
            ao_system_short_name=args.ao_system_short_name,
            config_short_name=args.config_short_name,
            build_root=args.build_root,
            aosky_conf=args.aosky_conf,
        )
    )
    return 0


def _handle_show(args: argparse.Namespace) -> int:
    from .build import show_build

    print(show_build(args.build))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
