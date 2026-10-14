from llm_router.core.tokens import (
    TOKENS_PER_MESSAGE,
    TOKENS_PER_PROMPT,
    estimate_messages,
    estimate_text,
)

# A typical English paragraph. Real tokenizers put this near 4 characters per token.
ENGLISH = (
    "The router sits between an application and its model providers. It decides which model "
    "handles each request, falls back when one fails, and refuses to spend past a daily cap."
)


def test_empty_text_is_zero():
    assert estimate_text("") == 0


def test_ascii_rounds_up_to_whole_tokens():
    assert estimate_text("a") == 1
    assert estimate_text("abc") == 1
    assert estimate_text("abcd") == 2


def test_errs_high_on_english():
    # Four characters per token is the usual figure; the estimate must not come in under it.
    assert estimate_text(ENGLISH) >= len(ENGLISH) / 4
    assert estimate_text(ENGLISH) == -(-len(ENGLISH) // 3)


def test_non_ascii_characters_count_as_whole_tokens():
    assert estimate_text("日本語") == 3
    assert estimate_text("naïve") == 2 + 1  # four ASCII letters, one accented letter
    assert estimate_text("👋🏽") == 2  # an emoji and its skin-tone modifier


def test_mixed_scripts_count_each_part_its_own_way():
    # Nine Cyrillic letters at one token each, plus ", " as two ASCII characters (one token).
    assert estimate_text("Привет, мир") == 9 + 1


def test_messages_add_overhead_per_message_and_per_prompt():
    assert estimate_messages([]) == TOKENS_PER_PROMPT
    assert estimate_messages(["abc"]) == TOKENS_PER_PROMPT + TOKENS_PER_MESSAGE + 1
    assert estimate_messages(["", ""]) == TOKENS_PER_PROMPT + 2 * TOKENS_PER_MESSAGE


def test_messages_accept_any_iterable():
    assert estimate_messages(text for text in ["abc", "def"]) == estimate_messages(["abc", "def"])
