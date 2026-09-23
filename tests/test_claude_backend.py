from decimal import Decimal

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from agent_hub.backends.claude import TOOL_SUMMARY_LIMIT, SessionTracker, summarize_tool_input
from agent_hub.domain import (
    AssistantText,
    Failed,
    Finished,
    SessionId,
    SessionStarted,
    ToolCall,
)


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
