"""Test the python function from src."""

import sys

sys.path.insert(0, "./src/")

from src.function import my_function


def test_function() -> None:
    """See it the function really returns true."""
    assert my_function() is True
