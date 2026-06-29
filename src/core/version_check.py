"""Python version check."""


def check_python_version(
    version: tuple[int, int],
    min_required: tuple[int, int] = (3, 11),
) -> bool:
    """Return True if version >= min_required (e.g. 3.11, 3.12, 3.13)."""
    return version >= min_required
