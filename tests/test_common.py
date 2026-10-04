from typing import Any

import pytest

from agent_hub.backends.common import (
    ToolError,
    ToolSuccess,
    deliver_file,
    parse_questions,
)
from agent_hub.domain import Allowed, Answered, Denied, OutgoingFile
from tests.fakes import ASK_INPUT, ASK_QUESTION, FakeChannel


def test_questions_are_parsed() -> None:
    assert parse_questions(ASK_INPUT) == (ASK_QUESTION,)


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


async def test_delivered_file_is_a_success() -> None:
    channel = FakeChannel(Allowed(), Answered(()))

    result = await deliver_file({"path": "out/report.pdf", "caption": "Отчёт"}, channel)

    assert channel.sent == [OutgoingFile("out/report.pdf", "Отчёт")]
    assert result == ToolSuccess("Файл out/report.pdf отправлен пользователю")


async def test_refused_delivery_is_a_tool_error() -> None:
    channel = FakeChannel(Allowed(), Answered(()), Denied("больше 50 МБ"))

    assert await deliver_file({"path": "big.zip"}, channel) == ToolError("больше 50 МБ")


@pytest.mark.parametrize("args", [{}, {"path": ""}, {"path": 1}, {"path": "a", "caption": 2}])
async def test_malformed_file_arguments_are_a_tool_error(args: dict[str, Any]) -> None:
    channel = FakeChannel(Allowed(), Answered(()))

    result = await deliver_file(args, channel)

    assert channel.sent == []
    assert isinstance(result, ToolError)
