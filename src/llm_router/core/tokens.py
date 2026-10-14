"""A deliberately conservative token estimate, used to size spending reservations.

The real count comes back from the provider after the call. Before the call, the router only
needs a number that is not lower than the real one, because a reservation that runs low lets a
request reserve less than it can cost. So this errs high on purpose:

- ASCII text counts one token per three characters. English averages closer to four, and code
  closer to three and a half, so this overestimates both.
- Every other character counts as a whole token. That is high for most European languages and
  close for Chinese and Japanese, where a character is often a token by itself.
- Each message adds a fixed overhead for its role and separators, and the prompt adds a little
  more for the reply's opening.

Erring high costs only headroom near the cap; settlement always uses the provider's real counts.
"""

from __future__ import annotations

from collections.abc import Iterable

ASCII_CHARS_PER_TOKEN = 3
TOKENS_PER_MESSAGE = 4
TOKENS_PER_PROMPT = 3


def estimate_text(text: str) -> int:
    """Estimated tokens in one piece of text."""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if char.isascii())
    other_chars = len(text) - ascii_chars
    return -(-ascii_chars // ASCII_CHARS_PER_TOKEN) + other_chars


def estimate_messages(contents: Iterable[str]) -> int:
    """Estimated prompt tokens for a conversation, given the text of each message in order."""
    total = TOKENS_PER_PROMPT
    for content in contents:
        total += TOKENS_PER_MESSAGE + estimate_text(content)
    return total
