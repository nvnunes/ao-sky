"""Minimal public package metadata helpers."""

PACKAGE_SUMMARY = (
    "ao-sky is a public Python package for Gaia-backed AO-sky mapping, with a "
    "canonical raw Gaia store, Gaia-domain proper-motion transforms, reusable "
    "spatial helpers, and in-memory asterism search APIs."
)


def describe_package() -> str:
    """Return the current public package summary."""

    return PACKAGE_SUMMARY
