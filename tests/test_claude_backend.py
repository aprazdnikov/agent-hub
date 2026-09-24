import base64
from collections.abc import Sequence
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
    TOOL_SUMMARY_LIMIT,
    SessionTracker,
    decide,
    parse_questions,
    summarize_tool_input,
    user_message,
)
from agent_hub.domain import (
    Allowed,
    Answered,
    AssistantText,
    Decision,
    Denied,
    Failed,
    Finished,
    Image,
    ImageMediaType,
    Prompt,
    Question,
    QuestionOption,
    QuestionsOutcome,
    SessionId,
    SessionStarted,
    ToolCall,
    ToolRequest,
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


_ASK_INPUT: dict[str, Any] = {
    "questions": [
        {
            "question": "Какой формат?",
            "header": "Формат",
            "options": [
                {"label": "Кратко", "description": "Только суть"},
                {"label": "Подробно", "description": "С примерами"},
            ],
            "multiSelect": False,
        }
    ]
}
_ASK_QUESTION = Question(
    "Какой формат?",
    "Формат",
    (QuestionOption("Кратко", "Только суть"), QuestionOption("Подробно", "С примерами")),
    multi_select=False,
)


class FakeGate:
    def __init__(self, decision: Decision, outcome: QuestionsOutcome) -> None:
        self.decision = decision
        self.outcome = outcome
        self.requests: list[ToolRequest] = []
        self.asked: list[Sequence[Question]] = []

    async def request(self, tool: ToolRequest) -> Decision:
        self.requests.append(tool)
        return self.decision

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome:
        self.asked.append(questions)
        return self.outcome


def test_questions_are_parsed() -> None:
    assert parse_questions(_ASK_INPUT) == (_ASK_QUESTION,)


@pytest.mark.parametrize(
    "tool_input",
    [
        {},
        {"questions": []},
        {"questions": "x"},
        {"questions": [{"header": "h", "options": []}]},
        {"questions": [{"question": "q", "options": [{"description": "no label"}]}]},
    ],
)
def test_malformed_questions_are_rejected(tool_input: dict[str, Any]) -> None:
    assert parse_questions(tool_input) is None


async def test_answered_questions_are_returned_as_tool_input() -> None:
    gate = FakeGate(Allowed(), Answered((("Какой формат?", "Кратко"),)))

    result = await decide("AskUserQuestion", _ASK_INPUT, gate)

    assert gate.asked == [(_ASK_QUESTION,)]
    assert gate.requests == []
    assert result == PermissionResultAllow(
        updated_input={**_ASK_INPUT, "answers": {"Какой формат?": "Кратко"}}
    )


async def test_declined_questions_deny_the_tool() -> None:
    gate = FakeGate(Allowed(), Denied("нет"))
    assert await decide("AskUserQuestion", _ASK_INPUT, gate) == PermissionResultDeny(message="нет")


async def test_malformed_question_falls_back_to_approval() -> None:
    gate = FakeGate(Denied("нет"), Answered(()))

    result = await decide("AskUserQuestion", {"questions": []}, gate)

    assert gate.asked == []
    assert [request.tool for request in gate.requests] == ["AskUserQuestion"]
    assert result == PermissionResultDeny(message="нет")


async def test_other_tools_go_through_approval() -> None:
    gate = FakeGate(Allowed(), Answered(()))

    assert await decide("Bash", {"command": "ls"}, gate) == PermissionResultAllow()
    assert gate.requests == [ToolRequest("Bash", "ls")]


def test_user_message_puts_images_before_text() -> None:
    prompt = Prompt("что на фото?", (Image(ImageMediaType.JPEG, b"\xff\xd8"),))

    assert user_message(prompt) == {
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
    }


def test_user_message_without_text_has_only_images() -> None:
    prompt = Prompt("", (Image(ImageMediaType.PNG, b"x"),))
    content = user_message(prompt)["message"]["content"]
    assert [block["type"] for block in content] == ["image"]


def test_question_tool_call_is_not_echoed() -> None:
    message = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="AskUserQuestion", input=_ASK_INPUT)], model="claude"
    )
    assert list(SessionTracker().translate(message)) == []
