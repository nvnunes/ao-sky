"""Thin command-line entrypoints for ao-sky."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from ._version import __version__
from .about import describe_package


def _add_runtime_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gaia-root", type=Path, default=None, help="Resolved Gaia root.")
    parser.add_argument("--build-root", type=Path, default=None, help="Resolved build root.")
    parser.add_argument("--model-root", type=Path, default=None, help="Resolved AO model root.")
    parser.add_argument(
        "--ao-sky-yaml",
        dest="aosky_yaml",
        type=Path,
        default=None,
        help="Optional ao-sky.yaml file with build.roots defaults.",
    )


def _add_gaia_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gaia-root", type=Path, default=None, help="Resolved Gaia root.")
    parser.add_argument(
        "--ao-sky-yaml",
        dest="aosky_yaml",
        type=Path,
        default=None,
        help="Optional ao-sky.yaml file with build.roots defaults.",
    )


def _add_traversal_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of Traversal worker processes to use.",
    )
    parser.add_argument(
        "--gaia-cache-entries",
        type=int,
        default=None,
        help="Worker-local Gaia table cache entry target; 0 disables caching.",
    )
    parser.add_argument(
        "--gaia-cache-mb",
        type=int,
        default=None,
        help="Worker-local Gaia table cache memory cap in MiB; 0 disables caching.",
    )
    parser.add_argument(
        "--parent-memory-limit-mb",
        type=int,
        default=None,
        help=(
            "Stop Traversal if total RAM exceeds this MiB limit; 0 disables the "
            "guard. YAML configs use build.memory_limit_mb."
        ),
    )
    parser.add_argument(
        "--telemetry",
        choices=("basic", "detailed"),
        default=None,
        help="Traversal telemetry level. 'detailed' writes per-pixel diagnostics.",
    )


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
        help="Create a new persisted build root from a merged build config YAML file.",
    )
    init_parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("ao-sky.yaml"),
        help="Build config YAML filename.",
    )
    _add_runtime_root_arguments(init_parser)
    init_parser.add_argument(
        "--survey-root",
        type=Path,
        default=None,
        help="Source survey root for builds with survey_overlays.",
    )
    init_parser.set_defaults(handler=_handle_init)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one initialized build to completion.",
    )
    run_parser.add_argument("build", type=Path, help="Build directory to run.")
    run_parser.add_argument(
        "--ao-sky-yaml",
        dest="aosky_yaml",
        type=Path,
        default=None,
        help="Optional ao-sky.yaml file with Traversal execution defaults.",
    )
    _add_traversal_execution_arguments(run_parser)
    run_parser.set_defaults(handler=_handle_run)

    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart the latest build version in a lineage workspace.",
    )
    restart_parser.add_argument("lineage_name", help="Build lineage name.")
    restart_parser.add_argument("--build-root", type=Path, default=None, help="Resolved build root.")
    _add_traversal_execution_arguments(restart_parser)
    restart_parser.add_argument(
        "--ao-sky-yaml",
        dest="aosky_yaml",
        type=Path,
        default=None,
        help="Optional ao-sky.yaml file with build root defaults.",
    )
    restart_parser.set_defaults(handler=_handle_restart)

    fetch_gaia_parser = subparsers.add_parser(
        "fetch-gaia",
        help="Install dust and materialize one full-sky canonical Gaia store.",
    )
    _add_gaia_root_arguments(fetch_gaia_parser)
    fetch_gaia_parser.add_argument(
        "--gaia-release",
        default=None,
        help="Gaia release identifier such as dr3.",
    )
    fetch_gaia_parser.add_argument(
        "--outer-level",
        default=None,
        type=int,
        help="Outer nested HEALPix level to materialize.",
    )
    fetch_gaia_parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh existing canonical Gaia files one pixel at a time.",
    )
    fetch_gaia_parser.set_defaults(handler=_handle_fetch_gaia)

    check_parser = subparsers.add_parser(
        "check",
        help="Validate the configured Gaia, build, dust, and model roots.",
    )
    _add_runtime_root_arguments(check_parser)
    check_parser.set_defaults(handler=_handle_check)

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

    build_path = init_build(
        config_filename=args.config,
        gaia_root=args.gaia_root,
        build_root=args.build_root,
        model_root=args.model_root,
        survey_root=args.survey_root,
        aosky_yaml=args.aosky_yaml,
    )
    print(build_path)
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    from .build import run_build

    print(
        run_build(
            args.build,
            workers=args.workers,
            gaia_cache_entries=args.gaia_cache_entries,
            gaia_cache_mb=args.gaia_cache_mb,
            parent_memory_limit_mb=args.parent_memory_limit_mb,
            telemetry=args.telemetry,
            aosky_yaml=args.aosky_yaml,
        )
    )
    return 0


def _handle_restart(args: argparse.Namespace) -> int:
    from .build import restart_build

    print(
        restart_build(
            lineage_name=args.lineage_name,
            build_root=args.build_root,
            aosky_yaml=args.aosky_yaml,
            workers=args.workers,
            gaia_cache_entries=args.gaia_cache_entries,
            gaia_cache_mb=args.gaia_cache_mb,
            parent_memory_limit_mb=args.parent_memory_limit_mb,
            telemetry=args.telemetry,
        )
    )
    return 0


def _handle_show(args: argparse.Namespace) -> int:
    from .build import show_build

    print(show_build(args.build))
    return 0


def _handle_fetch_gaia(args: argparse.Namespace) -> int:
    from .build import fetch_gaia_data
    from .build.config import load_build_definition

    gaia_release = args.gaia_release
    outer_level = args.outer_level
    if gaia_release is None or outer_level is None:
        definition, _ = load_build_definition(args.aosky_yaml or Path("ao-sky.yaml"))
        gaia_release = gaia_release or definition.gaia_release
        outer_level = outer_level if outer_level is not None else definition.outer_level

    print(
        fetch_gaia_data(
            gaia_root=args.gaia_root,
            gaia_release=gaia_release,
            outer_level=outer_level,
            force=args.force,
            aosky_yaml=args.aosky_yaml,
            output=sys.stdout,
        )
    )
    return 0


def _handle_check(args: argparse.Namespace) -> int:
    from .build import check_runtime_roots

    ok, report = check_runtime_roots(
        gaia_root=args.gaia_root,
        build_root=args.build_root,
        model_root=args.model_root,
        aosky_yaml=args.aosky_yaml,
    )
    print(report)
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
