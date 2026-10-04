"""Requests the Codex app-server sends the client: approvals, hub tools and questions."""

import json
from enum import StrEnum
from typing import Any, assert_never

from agent_hub.backends import UserChannel
from agent_hub.backends.codex_protocol import CodexTool
from agent_hub.backends.common import (
    TOOL_SUMMARY_LIMIT,
    HubTool,
    ToolError,
    ToolResult,
    ToolSuccess,
    deliver_file,
    parse_option,
    parse_questions,
)
from agent_hub.backends.rpc import JsonObject, MalformedRequestError, UnsupportedRequestError
from agent_hub.domain import Allowed, Answered, Denied, Question, ToolRequest
from agent_hub.render import truncate

_ASK_USER_SHAPE = (
    "questions must be a non-empty list of "
    "{question, header, options: [{label, description}], multiSelect}"
)


class ServerRequest(StrEnum):
    COMMAND_APPROVAL = "item/commandExecution/requestApproval"
    FILE_CHANGE_APPROVAL = "item/fileChange/requestApproval"
    TOOL_CALL = "item/tool/call"
    USER_INPUT = "item/tool/requestUserInput"


async def answer(channel: UserChannel, method: str, params: JsonObject) -> JsonObject:
    """Reply to one app-server request on behalf of the human behind `channel`."""
    try:
        request = ServerRequest(method)
    except ValueError as error:
        raise UnsupportedRequestError(method) from error
    match request:
        case ServerRequest.COMMAND_APPROVAL:
            summary = truncate(_command_summary(params), TOOL_SUMMARY_LIMIT)
            return await _approve(channel, ToolRequest(CodexTool.SHELL, summary))
        case ServerRequest.FILE_CHANGE_APPROVAL:
            summary = truncate(_file_change_summary(params), TOOL_SUMMARY_LIMIT)
            return await _approve(channel, ToolRequest(CodexTool.PATCH, summary))
        case ServerRequest.TOOL_CALL:
            return _tool_response(await _call_tool(channel, params))
        case ServerRequest.USER_INPUT:
            return await _user_input(channel, params)
        case _:
            assert_never(request)


def _command_summary(params: JsonObject) -> str:
    match params:
        case {"command": str() as command} if command.strip():
            return command
        case {"reason": str() as reason} if reason.strip():
            return reason
        case _:
            return json.dumps(params, ensure_ascii=False)


def _file_change_summary(params: JsonObject) -> str:
    # The changed paths were already shown as the `patch` line of the item.
    match params:
        case {"reason": str() as reason} if reason.strip():
            return reason
        case {"grantRoot": str() as root}:
            return f"запись в {root}"
        case _:
            return "изменение файлов"


async def _approve(channel: UserChannel, request: ToolRequest) -> JsonObject:
    decision = await channel.request(request)
    match decision:
        case Allowed():
            return {"decision": "accept"}
        case Denied():
            return {"decision": "decline"}
        case _:
            assert_never(decision)


async def _call_tool(channel: UserChannel, params: JsonObject) -> ToolResult:
    match params:
        case {"tool": HubTool.SEND_FILE, "arguments": dict() as arguments}:
            return await deliver_file(arguments, channel)
        case {"tool": HubTool.ASK_USER, "arguments": dict() as arguments}:
            return await _ask(channel, arguments)
        case {"tool": str() as tool}:
            return ToolError(f"unknown tool {tool} or its arguments are not an object")
        case _:
            raise MalformedRequestError(f"tool call without a tool name: {params!r}")


async def _ask(channel: UserChannel, arguments: dict[str, Any]) -> ToolResult:
    questions = parse_questions(arguments)
    if questions is None:
        return ToolError(_ASK_USER_SHAPE)
    outcome = await channel.ask(questions)
    match outcome:
        case Answered(answers):
            return ToolSuccess("\n".join(f"{question}: {reply}" for question, reply in answers))
        case Denied(reason):
            return ToolError(reason)
        case _:
            assert_never(outcome)


def _tool_response(result: ToolResult) -> JsonObject:
    match result:
        case ToolSuccess(text):
            return {"success": True, "contentItems": [{"type": "inputText", "text": text}]}
        case ToolError(text):
            return {"success": False, "contentItems": [{"type": "inputText", "text": text}]}
        case _:
            assert_never(result)


async def _user_input(channel: UserChannel, params: JsonObject) -> JsonObject:
    asked = _user_questions(params)
    outcome = await channel.ask(tuple(question for _, question in asked))
    match outcome:
        case Answered(answers):
            replies = dict(answers)
            return {
                "answers": {
                    question_id: {"answers": [replies[question.text]]}
                    for question_id, question in asked
                    if question.text in replies
                }
            }
        case Denied():
            # Codex goes on with its own judgment when a question stays unanswered.
            return {"answers": {}}
        case _:
            assert_never(outcome)


def _user_questions(params: JsonObject) -> tuple[tuple[str, Question], ...]:
    match params:
        case {"questions": list() as raw} if raw:
            return tuple(_user_question(item) for item in raw)
        case _:
            raise MalformedRequestError(f"requestUserInput without questions: {params!r}")


def _user_question(raw: object) -> tuple[str, Question]:
    match raw:
        case {
            "id": str() as question_id,
            "header": str() as header,
            "question": str() as text,
            "options": list() as options,
        } if text.strip():
            return question_id, _question(text, header, options)
        case {"id": str() as question_id, "header": str() as header, "question": str() as text} if (
            text.strip()
        ):
            return question_id, _question(text, header, [])
        case _:
            raise MalformedRequestError(f"malformed question: {raw!r}")


def _question(text: str, header: str, raw_options: list[Any]) -> Question:
    parsed = [parse_option(item) for item in raw_options]
    options = tuple(option for option in parsed if option is not None)
    if len(options) != len(parsed):
        raise MalformedRequestError(f"malformed options: {raw_options!r}")
    return Question(text, header, options, multi_select=False)
