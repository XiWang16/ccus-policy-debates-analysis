"""
Manually curated list of CCUS-related bills.

Each entry is either:
  - "C-NNN"          — process all sessions for this bill number
  - "C-NNN/SS-S"     — process only the specified session (e.g. "C-50/44-1")
  - "S-NNN/SS-S"     — Senate bill pinned to a session

The pipeline checks each bill's full text and speeches for CCUS keywords
and prints a warning if none are found — a sanity check that the bill is
genuinely CCUS-related.
"""
import re

# ---------------------------------------------------------------------------
# Running list of manually identified CCUS-related bills
# ---------------------------------------------------------------------------
MANUAL_CCUS_BILL_NUMBERS: list[str] = [
    "S-243/44-1",  # Climate-Aligned Finance Act
    "C-262/43-2",  # An Act to amend the Income Tax Act (capture and utilization or storage of carbon dioxide)
    "C-19/44-1",   # Budget Implementation Act, 2022, No. 1 (introduced CCUS Investment Tax Credit)
    "C-50/39-2",   # Budget Implementation Act 2007 (contains CCUS provisions)
    "C-59/44-1",   # Fall Economic Statement Implementation Act 2023 (CCUS ITC provisions)
    "C-69/42-1",   # Impact Assessment Act
]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
_BILL_NUMBER_RE = re.compile(r"^[SC]-\d+$")
_BILL_ENTRY_RE = re.compile(r"^([SC]-\d+)(?:/(\d+-\d+))?$")


def parse_bill_entry(entry: str) -> tuple[str, str | None]:
    """Parse a bill entry into (number, session_or_None).

    Returns ``(number, session)`` where *session* may be ``None`` if no
    session was specified.  Returns ``(entry, None)`` with a warning printed
    if the format is unrecognised.
    """
    m = _BILL_ENTRY_RE.match(entry.strip())
    if not m:
        print(
            f"[Manual bills] WARNING: '{entry}' is not a valid bill entry "
            "(expected C-NNN or C-NNN/SS-S). Skipping."
        )
        return (entry, None)
    return (m.group(1), m.group(2))


def is_valid_bill_number(number: str) -> bool:
    """Return True if *number* follows the expected S-NNN or C-NNN format."""
    return bool(_BILL_NUMBER_RE.match(number.strip()))


def get_manual_bill_entries() -> list[tuple[str, str | None]]:
    """Return a list of ``(number, session_or_None)`` tuples from the manual list.

    Invalid entries are skipped with a warning.
    """
    result: list[tuple[str, str | None]] = []
    for raw in MANUAL_CCUS_BILL_NUMBERS:
        number, session = parse_bill_entry(raw)
        if is_valid_bill_number(number):
            result.append((number, session))
    return result


def get_manual_bill_numbers() -> list[str]:
    """Return just the bill numbers (without session pins) for backward compat."""
    return [number for number, _ in get_manual_bill_entries()]
