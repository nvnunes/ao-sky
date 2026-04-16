"""Explicit environment-preflight helpers for CLI root checks."""

from __future__ import annotations

from pathlib import Path

from ..dust import fetch_gaia_tge_dataset, gaia_tge_map_filename
from .config import resolve_dust_root_only, resolve_runtime_root_candidates


def fetch_dust_data(
    *,
    dust_root: Path | None,
    aosky_conf: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve `dust_root`, fetch Gaia TGE into it, and return the dataset filename."""

    resolved_dust_root = resolve_dust_root_only(
        dust_root=dust_root,
        aosky_conf=aosky_conf,
        cwd=cwd,
    )
    return fetch_gaia_tge_dataset(resolved_dust_root)


def check_runtime_roots(
    *,
    gaia_root: Path | None,
    build_root: Path | None,
    dust_root: Path | None,
    model_root: Path | None,
    aosky_conf: Path | None = None,
    cwd: Path | None = None,
) -> tuple[bool, str]:
    """Return one human-readable root report plus an overall success flag."""

    candidates = resolve_runtime_root_candidates(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=dust_root,
        model_root=model_root,
        aosky_conf=aosky_conf,
        cwd=cwd,
    )

    lines: list[str] = []
    overall_ok = True
    for name in ("gaia_root", "build_root", "dust_root", "model_root"):
        path = candidates[name]
        ok = _root_ok(name, path)
        overall_ok = overall_ok and ok
        rendered_path = "<unset>" if path is None else str(path)
        status = "OK" if ok else "MISSING"
        lines.append(f"{name}: {status} {rendered_path}")

    return overall_ok, "\n".join(lines)


def _root_ok(name: str, path: Path | None) -> bool:
    if path is None:
        return False
    if name == "dust_root":
        try:
            gaia_tge_map_filename(path)
        except Exception:
            return False
        return True
    return path.exists()
