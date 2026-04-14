"""Minimal public package metadata helpers."""

PACKAGE_SUMMARY = (
    "ao-sky is in its bootstrap stage. The installable package, CLI surface, "
    "and repo-local docs workflow are in place; Gaia, spatial, and asterism "
    "implementations are the next planned areas of work."
)


def describe_package() -> str:
    """Return the current public package summary."""

    return PACKAGE_SUMMARY
