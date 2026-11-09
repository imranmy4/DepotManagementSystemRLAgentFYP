"""ISO 6346 container number validation (format + check digit).

A number is 3 owner letters, a category letter (U freight, J detachable equipment,
Z trailer/chassis), a 6-digit serial and a check digit, e.g. CSQU3054383.
"""
import re

_FORMAT = re.compile(r"^[A-Z]{3}[UJZ][0-9]{7}$")

# Letter values start at A=10 and skip multiples of 11 (11, 22, 33).
_LETTER_VALUES = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                          (v for v in range(10, 39) if v % 11)))


def check_digit(first10: str) -> int:
    """Check digit for the first 10 characters (owner, category, serial)."""
    total = sum((_LETTER_VALUES[c] if c.isalpha() else int(c)) * 2 ** i
                for i, c in enumerate(first10))
    return total % 11 % 10


def is_valid(container_no: str) -> bool:
    return bool(_FORMAT.match(container_no)) and \
        check_digit(container_no[:10]) == int(container_no[10])

