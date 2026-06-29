"""Tests for Python version check."""

from src.core.version_check import check_python_version


def test_311_ok():
    assert check_python_version((3, 11))


def test_313_ok():
    assert check_python_version((3, 13))


def test_310_fails():
    assert not check_python_version((3, 10))
