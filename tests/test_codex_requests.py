from typing import Any

import pytest

from agent_hub.backends.codex_requests import answer
from agent_hub.backends.rpc import MalformedRequestError, UnsupportedRequestError
from agent_hub.domain import (
    Allowed,
    Answered,
    Denied,
    OutgoingFile,
    Question,
    QuestionOption,
    ToolRequest,
)
from tests.fakes import ASK_INPUT, ASK_QUESTION, FakeChannel

COMMAND = "item/commandExecution/requestApproval"
FILE_CHANGE = "item/fileChange/requestApproval"
TOOL_CALL = "item/tool/call"
USER_INPUT = "item/tool/requestUserInput"
_BASE = {"threadId": "th-1", "turnId": "turn-1", "itemId": "i-1", "startedAtMs": 0}


ALLOWED = Allowed()


def _channel(decision: Allowed | Denied = ALLOWED) -> FakeChannel:
    return FakeChannel(decision, Answered((("Какой формат?", "Кратко"),)))


@pytest.mark.parametrize(
    ("decision", "expected"), [(Allowed(), "accept"), (Denied("нет"), "decline")]
)
async def test_command_approval_follows_the_user(decision: Allowed | Denied, expected: str) -> None:
    channel = _channel(decision)

    reply = await answer(channel, COMMAND, {**_BASE, "command": "npm test", "cwd": "/w"})

    assert reply == {"decision": expected}
    assert channel.requests == [ToolRequest("shell", "npm test")]


async def test_command_approval_without_command_shows_the_reason() -> None:
    channel = _channel()
    await answer(channel, COMMAND, {**_BASE, "command": None, "reason": "нужна сеть"})
    assert channel.requests == [ToolRequest("shell", "нужна сеть")]


@pytest.mark.parametrize(
    ("extra", "summary"),
    [
        ({"reason": "правка вне песочницы"}, "правка вне песочницы"),
        ({"grantRoot": "/etc"}, "запись в /etc"),
        ({}, "изменение файлов"),
    ],
)
async def test_file_change_approval(extra: dict[str, Any], summary: str) -> None:
    channel = _channel()

    assert await answer(channel, FILE_CHANGE, {**_BASE, **extra}) == {"decision": "accept"}
    assert channel.requests == [ToolRequest("patch", summary)]


async def test_send_file_tool_delivers() -> None:
    channel = _channel()
    params = {**_BASE, "callId": "c", "tool": "send_file", "arguments": {"path": "r.pdf"}}

    reply = await answer(channel, TOOL_CALL, params)

    assert channel.sent == [OutgoingFile("r.pdf", "")]
    assert reply == {
        "success": True,
        "contentItems": [{"type": "inputText", "text": "Файл r.pdf отправлен пользователю"}],
    }


async def test_refused_file_is_an_unsuccessful_tool_call() -> None:
    channel = FakeChannel(Allowed(), Answered(()), Denied("вне директории"))
    params = {**_BASE, "callId": "c", "tool": "send_file", "arguments": {"path": "/etc/x"}}

    reply = await answer(channel, TOOL_CALL, params)

    assert reply == {
        "success": False,
        "contentItems": [{"type": "inputText", "text": "вне директории"}],
    }


async def test_ask_user_tool_returns_answers_as_text() -> None:
    channel = _channel()
    params = {**_BASE, "callId": "c", "tool": "ask_user", "arguments": ASK_INPUT}

    reply = await answer(channel, TOOL_CALL, params)

    assert channel.asked == [(ASK_QUESTION,)]
    assert reply["success"] is True
    assert reply["contentItems"][0]["text"] == "Какой формат?: Кратко"


async def test_malformed_ask_user_is_an_unsuccessful_tool_call() -> None:
    channel = _channel()
    params = {**_BASE, "callId": "c", "tool": "ask_user", "arguments": {"questions": []}}

    reply = await answer(channel, TOOL_CALL, params)

    assert channel.asked == []
    assert reply["success"] is False


async def test_unknown_dynamic_tool_is_an_unsuccessful_tool_call() -> None:
    params = {**_BASE, "callId": "c", "tool": "rm_rf", "arguments": {}}
    assert (await answer(_channel(), TOOL_CALL, params))["success"] is False


async def test_tool_call_without_a_name_is_malformed() -> None:
    with pytest.raises(MalformedRequestError):
        await answer(_channel(), TOOL_CALL, {**_BASE, "arguments": {}})


async def test_native_questions_are_answered_by_id() -> None:
    channel = _channel()
    params = {
        **_BASE,
        "isBlocking": True,
        "questions": [
            {
                "id": "q1",
                "header": "Формат",
                "question": "Какой формат?",
                "options": [{"label": "Кратко", "description": "Только суть"}],
            }
        ],
    }

    reply = await answer(channel, USER_INPUT, params)

    assert channel.asked == [
        (
            Question(
                "Какой формат?",
                "Формат",
                (QuestionOption("Кратко", "Только суть"),),
                multi_select=False,
            ),
        )
    ]
    assert reply == {"answers": {"q1": {"answers": ["Кратко"]}}}


async def test_declined_native_questions_get_no_answers() -> None:
    channel = FakeChannel(Allowed(), Denied("не отвечу"))
    params = {
        **_BASE,
        "isBlocking": True,
        "questions": [{"id": "q1", "header": "", "question": "Да?"}],
    }

    assert await answer(channel, USER_INPUT, params) == {"answers": {}}


@pytest.mark.parametrize(
    "questions",
    [
        [],
        [{"id": "q1", "header": "h"}],
        [{"id": "q1", "header": "h", "question": "?", "options": [{}]}],
    ],
)
async def test_malformed_native_questions(questions: list[Any]) -> None:
    with pytest.raises(MalformedRequestError):
        await answer(_channel(), USER_INPUT, {**_BASE, "isBlocking": True, "questions": questions})


async def test_unknown_request_is_unsupported() -> None:
    with pytest.raises(UnsupportedRequestError):
        await answer(_channel(), "mcpServer/elicitation/request", {})
