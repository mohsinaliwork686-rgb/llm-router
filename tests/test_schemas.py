import pytest
from pydantic import ValidationError

from llm_router.core.chat import ChatMessage
from llm_router.schemas import ChatCompletionRequest


def parse(**body):
    return ChatCompletionRequest.model_validate(
        {"model": "cheap", "messages": [{"role": "user", "content": "Hi"}], **body}
    )


def test_developer_messages_become_system_messages():
    request = parse(
        messages=[{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    ).to_chat_request()
    assert request.messages == (ChatMessage("system", "Be brief."), ChatMessage("user", "Hi"))


def test_text_parts_are_joined():
    request = parse(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First."},
                    {"type": "text", "text": "Second."},
                ],
            }
        ]
    ).to_chat_request()
    assert request.messages[0].content == "First.\nSecond."


def test_null_content_is_empty_text():
    request = parse(
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": None}]
    ).to_chat_request()
    assert request.messages[1].content == ""


def test_max_completion_tokens_wins_over_max_tokens():
    assert parse(max_tokens=10, max_completion_tokens=20).to_chat_request().max_tokens == 20
    assert parse(max_tokens=10).to_chat_request().max_tokens == 10
    assert parse().to_chat_request().max_tokens is None


def test_stop_accepts_a_string_or_a_list():
    assert parse(stop="END").to_chat_request().stop == ("END",)
    assert parse(stop=["A", "B"]).to_chat_request().stop == ("A", "B")
    assert parse().to_chat_request().stop == ()


def test_more_than_four_stop_sequences_is_refused():
    with pytest.raises(ValidationError, match="at most 4"):
        parse(stop=["a", "b", "c", "d", "e"])


def test_sampling_params_pass_through():
    request = parse(temperature=0.5, top_p=0.8).to_chat_request()
    assert (request.temperature, request.top_p) == (0.5, 0.8)


def test_assistant_tool_calls_are_refused():
    with pytest.raises(ValidationError, match="tool calls are not supported"):
        parse(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            ]
        )


def test_message_names_are_ignored():
    request = parse(messages=[{"role": "user", "content": "Hi", "name": "sam"}]).to_chat_request()
    assert request.messages == (ChatMessage("user", "Hi"),)
