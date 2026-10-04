import base64
from decimal import Decimal
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from agent_hub.backends.claude import (
    SEND_FILE_TOOL,
    SessionTracker,
    decide,
    send_file_result,
    summarize_tool_input,
    user_message,
)
from agent_hub.backends.common import TOOL_SUMMARY_LIMIT
from agent_hub.domain import (
    Allowed,
    Answered,
    AssistantText,
    Denied,
    Failed,
    Finished,
    Image,
    ImageMediaType,
    OutgoingFile,
    Prompt,
    SessionId,
    SessionStarted,
    ToolCall,
    ToolRequest,
)
from tests.fakes import ASK_INPUT, ASK_QUESTION, FakeChannel


def _result(*, is_error: bool, cost: float | None = 0.1234) -> ResultMessage:
    return ResultMessage(
        subtype="error_during_execution" if is_error else "success",
        duration_ms=10,
        duration_api_ms=5,
        is_error=is_error,
        num_turns=3,
        session_id="s-1",
        total_cost_usd=cost,
        result="boom" if is_error else "ok",
    )


def test_session_start_is_reported_once() -> None:
    tracker = SessionTracker()
    init = SystemMessage(subtype="init", data={"session_id": "s-1"})

    assert list(tracker.translate(init)) == [SessionStarted(SessionId("s-1"))]
    assert list(tracker.translate(init)) == []


def test_assistant_blocks_become_text_and_tool_calls() -> None:
    message = AssistantMessage(
        content=[
            TextBlock(text="Смотрю тесты"),
            TextBlock(text="   "),
            ToolUseBlock(id="t1", name="Bash", input={"command": "uv run pytest"}),
        ],
        model="claude",
    )

    assert list(SessionTracker().translate(message)) == [
        AssistantText("Смотрю тесты"),
        ToolCall("Bash", "uv run pytest"),
    ]


def test_success_result_finishes_turn() -> None:
    tracker = SessionTracker()
    events = list(tracker.translate(_result(is_error=False)))

    assert events == [
        SessionStarted(SessionId("s-1")),
        Finished(SessionId("s-1"), turns=3, cost_usd=Decimal("0.1234")),
    ]
    assert tracker.terminated


def test_missing_cost_is_none() -> None:
    events = list(SessionTracker().translate(_result(is_error=False, cost=None)))
    assert events[-1] == Finished(SessionId("s-1"), turns=3, cost_usd=None)


def test_error_result_fails_turn() -> None:
    tracker = SessionTracker()
    events = list(tracker.translate(_result(is_error=True)))

    assert events[-1] == Failed("error_during_execution: boom")
    assert tracker.terminated


def test_known_tool_is_summarized_by_its_main_argument() -> None:
    assert summarize_tool_input("Read", {"file_path": "/a/b.py", "limit": 10}) == "/a/b.py"


def test_unknown_tool_is_summarized_as_json() -> None:
    assert summarize_tool_input("mcp__x", {"q": "привет"}) == '{"q": "привет"}'


def test_summary_is_truncated() -> None:
    summary = summarize_tool_input("Bash", {"command": "x" * (TOOL_SUMMARY_LIMIT * 2)})
    assert len(summary) == TOOL_SUMMARY_LIMIT


async def test_answered_questions_are_returned_as_tool_input() -> None:
    channel = FakeChannel(Allowed(), Answered((("Какой формат?", "Кратко"),)))

    result = await decide("AskUserQuestion", ASK_INPUT, channel)

    assert channel.asked == [(ASK_QUESTION,)]
    assert channel.requests == []
    assert result == PermissionResultAllow(
        updated_input={**ASK_INPUT, "answers": {"Какой формат?": "Кратко"}}
    )


async def test_declined_questions_deny_the_tool() -> None:
    channel = FakeChannel(Allowed(), Denied("нет"))
    result = await decide("AskUserQuestion", ASK_INPUT, channel)
    assert result == PermissionResultDeny(message="нет")


async def test_malformed_question_falls_back_to_approval() -> None:
    channel = FakeChannel(Denied("нет"), Answered(()))

    result = await decide("AskUserQuestion", {"questions": []}, channel)

    assert channel.asked == []
    assert [request.tool for request in channel.requests] == ["AskUserQuestion"]
    assert result == PermissionResultDeny(message="нет")


async def test_other_tools_go_through_approval() -> None:
    channel = FakeChannel(Allowed(), Answered(()))

    assert await decide("Bash", {"command": "ls"}, channel) == PermissionResultAllow()
    assert channel.requests == [ToolRequest("Bash", "ls")]


def test_user_message_puts_images_before_text() -> None:
    prompt = Prompt("что на фото?", (Image(ImageMediaType.JPEG, b"\xff\xd8"),))

    assert user_message(prompt, "p1") == {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(b"\xff\xd8").decode(),
                    },
                },
                {"type": "text", "text": "что на фото?"},
            ],
        },
        "parent_tool_use_id": None,
        "uuid": "p1",
    }


def test_user_message_without_text_has_only_images() -> None:
    prompt = Prompt("", (Image(ImageMediaType.PNG, b"x"),))
    content = user_message(prompt, "p1")["message"]["content"]
    assert [block["type"] for block in content] == ["image"]


def test_question_tool_call_is_not_echoed() -> None:
    message = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="AskUserQuestion", input=ASK_INPUT)], model="claude"
    )
    assert list(SessionTracker().translate(message)) == []


async def test_send_file_is_allowed_without_asking() -> None:
    channel = FakeChannel(Denied("нет"), Answered(()))

    assert await decide(SEND_FILE_TOOL, {"path": "a.pdf"}, channel) == PermissionResultAllow()
    assert channel.requests == []


async def test_send_file_delivers_and_reports_success() -> None:
    channel = FakeChannel(Allowed(), Answered(()))

    result = await send_file_result({"path": "out/report.pdf", "caption": "Отчёт"}, channel)

    assert channel.sent == [OutgoingFile("out/report.pdf", "Отчёт")]
    assert result.get("is_error") is not True
    assert "report.pdf" in result["content"][0]["text"]


async def test_send_file_failure_is_a_tool_error() -> None:
    channel = FakeChannel(Allowed(), Answered(()), Denied("больше 50 МБ"))

    result = await send_file_result({"path": "big.zip"}, channel)

    assert channel.sent == [OutgoingFile("big.zip", "")]
    assert result["is_error"] is True
    assert result["content"][0]["text"] == "больше 50 МБ"


@pytest.mark.parametrize("args", [{}, {"path": ""}, {"path": 1}, {"path": "a", "caption": 2}])
async def test_send_file_rejects_malformed_arguments(args: dict[str, Any]) -> None:
    channel = FakeChannel(Allowed(), Answered(()))

    result = await send_file_result(args, channel)

    assert channel.sent == []
    assert result["is_error"] is True


def test_send_file_call_is_not_echoed() -> None:
    message = AssistantMessage(
        content=[ToolUseBlock(id="t1", name=SEND_FILE_TOOL, input={"path": "a"})], model="claude"
    )
    assert list(SessionTracker().translate(message)) == []
