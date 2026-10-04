"""Hub tools and input parsing shared by agent backends."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, assert_never, final

from agent_hub.backends import UserChannel
from agent_hub.domain import Delivered, Denied, OutgoingFile, Question, QuestionOption

TOOL_SUMMARY_LIMIT = 600


class HubTool(StrEnum):
    """Tools agent-hub itself gives the agent."""

    SEND_FILE = "send_file"
    ASK_USER = "ask_user"


SEND_FILE_DESCRIPTION = (
    "Send a file to the user in their Telegram chat. The user only sees your text replies, "
    "so use this whenever they ask for a file or a file is the natural result (a PDF report, "
    "an archive, an image, a CSV export). `path` is absolute or relative to the working "
    "directory and must stay inside it; `caption` is optional text shown under the file."
)
SEND_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "caption": {"type": "string"}},
    "required": ["path"],
}
ASK_USER_DESCRIPTION = (
    "Ask the user clarifying questions in their Telegram chat and wait for the answers. Use it "
    "only when an answer would materially change the work. Each question has a short header "
    "and up to four options the user picks with a button; the user may also reply with free "
    "text. Set multiSelect to let the user pick several options."
)
ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "header": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["label"],
                        },
                    },
                    "multiSelect": {"type": "boolean"},
                },
                "required": ["question"],
            },
        }
    },
    "required": ["questions"],
}


@final
@dataclass(frozen=True, slots=True)
class ToolSuccess:
    text: str


@final
@dataclass(frozen=True, slots=True)
class ToolError:
    text: str


ToolResult = ToolSuccess | ToolError


async def deliver_file(args: Mapping[str, Any], channel: UserChannel) -> ToolResult:
    """`send_file` tool body: a failed delivery is a tool error the agent can react to."""
    path, caption = args.get("path"), args.get("caption", "")
    if not isinstance(path, str) or not path.strip() or not isinstance(caption, str):
        return ToolError("path must be a non-empty string, caption a string")
    delivery = await channel.send_file(OutgoingFile(path, caption))
    match delivery:
        case Delivered():
            return ToolSuccess(f"Файл {path} отправлен пользователю")
        case Denied(reason):
            return ToolError(reason)
        case _:
            assert_never(delivery)


def parse_questions(tool_input: Mapping[str, Any]) -> tuple[Question, ...] | None:
    """Questions input → questions; None when it does not match the expected shape."""
    raw = tool_input.get("questions")
    if not isinstance(raw, list) or not raw:
        return None
    questions: list[Question] = []
    for item in raw:
        question = _parse_question(item)
        if question is None:
            return None
        questions.append(question)
    return tuple(questions)


def _parse_question(raw: object) -> Question | None:
    if not isinstance(raw, dict):
        return None
    text, header, options = raw.get("question"), raw.get("header", ""), raw.get("options", [])
    if not isinstance(text, str) or not text.strip() or not isinstance(header, str):
        return None
    if not isinstance(options, list):
        return None
    parsed: list[QuestionOption] = []
    for item in options:
        option = parse_option(item)
        if option is None:
            return None
        parsed.append(option)
    return Question(text, header, tuple(parsed), multi_select=raw.get("multiSelect") is True)


def parse_option(raw: object) -> QuestionOption | None:
    if not isinstance(raw, dict):
        return None
    label, description = raw.get("label"), raw.get("description", "")
    if not isinstance(label, str) or not label.strip() or not isinstance(description, str):
        return None
    return QuestionOption(label, description)
