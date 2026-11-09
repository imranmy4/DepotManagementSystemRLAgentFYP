"""Tests for ISO 6346 check-digit validation."""
import pytest

from app import iso6346


@pytest.mark.parametrize("no", ["CSQU3054383", "MSKU9070323", "ABCU1234560"])
def test_valid_numbers(no):
    assert iso6346.is_valid(no)


@pytest.mark.parametrize("no", ["CSQU3054384", "CSQX3054383", "CSQU305438", "csqu3054383", ""])
def test_invalid_numbers(no):
    assert not iso6346.is_valid(no)


def test_check_digit_ten_maps_to_zero():
    # Remainder 10 is written as 0.
    first10 = next(f"DPTU{n:06d}" for n in range(1_000_000)
                   if sum((iso6346._LETTER_VALUES[c] if c.isalpha() else int(c)) * 2 ** i
                          for i, c in enumerate(f"DPTU{n:06d}")) % 11 == 10)
    assert iso6346.check_digit(first10) == 0

