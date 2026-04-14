"""Minimal public package metadata helpers."""

PACKAGE_SUMMARY = (
    "ao-sky now includes its raw Gaia store boundary alongside the installable "
    "package, CLI surface, and repo-local docs workflow; spatial, asterism, "
    "and derived photometric layers remain planned work."
)


def describe_package() -> str:
    """Return the current public package summary."""

    return PACKAGE_SUMMARY
