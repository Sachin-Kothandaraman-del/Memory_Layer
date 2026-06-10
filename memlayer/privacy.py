"""Local-first privacy guards: PII redaction and never-remember detection.

Everything here is pure regex — no text ever leaves the machine for privacy
checks. Redaction (when enabled) runs BEFORE embedding/extraction, so PII is
never sent to the Gemini API, never embedded, and never stored.
"""

from __future__ import annotations

import re

# Ordered: more specific patterns first so e.g. an SSN isn't half-eaten by
# the phone pattern.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{6,}\d(?![\w])")
_IP = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


def _digits(s: str) -> int:
    return sum(c.isdigit() for c in s)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace common PII with typed placeholders.

    Returns (clean_text, counts_by_type). Digit-count guards keep dates and
    version numbers from being mistaken for phone numbers or cards.
    """
    counts: dict[str, int] = {}

    def sub(name: str, pattern: re.Pattern, s: str, min_digits: int = 0) -> str:
        def repl(m: re.Match) -> str:
            if min_digits and _digits(m.group()) < min_digits:
                return m.group()
            counts[name] = counts.get(name, 0) + 1
            return f"[REDACTED_{name.upper()}]"

        return pattern.sub(repl, s)

    text = sub("email", _EMAIL, text)
    text = sub("ssn", _SSN, text)
    text = sub("credit_card", _CARD, text, min_digits=13)
    text = sub("phone", _PHONE, text, min_digits=9)
    text = sub("ip_address", _IP, text)
    return text, counts


# Phrases that mean "do not store this". Matching input is not embedded,
# not extracted, and not persisted — only an audit entry records the skip.
DEFAULT_NEVER_REMEMBER: tuple[str, ...] = (
    r"\boff the record\b",
    r"\bdon'?t\s+(?:remember|save|store|log)\s+(?:this|that|any of this)\b",
    r"\bdo\s+not\s+(?:remember|save|store|log)\s+(?:this|that)\b",
    r"\bforget\s+(?:i|I)\s+(?:said|asked|mentioned)\b",
)


def matches_never_remember(text: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)
