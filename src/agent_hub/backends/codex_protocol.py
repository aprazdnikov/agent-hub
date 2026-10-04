"""Pure mapping between Codex app-server messages and agent-hub types."""

import base64
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, assert_never, final

from agent_hub.backends.common import (
    ASK_USER_DESCRIPTION,
    ASK_USER_SCHEMA,
    SEND_FILE_DESCRIPTION,
    SEND_FILE_SCHEMA,
    TOOL_SUMMARY_LIMIT,
    HubTool,
)
from agent_hub.backends.rpc import JsonObject, Notification, ProtocolError
from agent_hub.config import CodexSettings
from agent_hub.domain import (
    AgentEvent,
    AssistantText,
    Failed,
    Finished,
    Image,
    Prompt,
    SessionId,
    ToolCall,
    TopicSession,
)
from agent_hub.render import truncate

TurnId = NewType("TurnId", str)

CLIENT_INFO: JsonObject = {"name": "agent-hub", "title": "agent-hub", "version": "0.1.0"}
# Approvals go to the human in Telegram, never to Codex's own reviewer agent.
APPROVALS_REVIEWER = "user"
UNNAMED_FILE_CHANGE = "изменение файлов"


class Call(StrEnum):
    """Client-to-server methods agent-hub uses."""

    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    ACCOUNT_READ = "account/read"
    LOGIN = "account/login/start"
    THREAD_START = "thread/start"
    THREAD_RESUME = "thread/resume"
    TURN_START = "turn/start"
    TURN_STEER = "turn/steer"


class Notice(StrEnum):
    """Server notifications a session reacts to; all others are ignored."""

    ITEM_STARTED = "item/started"
    ITEM_COMPLETED = "item/completed"
    TOKEN_USAGE = "thread/tokenUsage/updated"  # noqa: S105 - a notification name, not a secret
    TURN_COMPLETED = "turn/completed"


class CodexTool(StrEnum):
    """Names under which Codex's built-in actions appear in the topic."""

    SHELL = "shell"
    PATCH = "patch"
    WEB_SEARCH = "web_search"


class CodexAuth(StrEnum):
    API_KEY = "apiKey"
    CHATGPT = "chatgpt"
    OTHER = "other"
    NOT_REQUIRED = "notRequired"
    MISSING = "missing"


@final
@dataclass(frozen=True, slots=True)
class Translation:
    events: tuple[AgentEvent, ...] = ()
    completed: TurnId | None = None


def initialize_params() -> JsonObject:
    # Dynamic tools are part of the experimental app-server API.
    return {"clientInfo": CLIENT_INFO, "capabilities": {"experimentalApi": True}}


def api_key_login_params(api_key: str) -> JsonObject:
    return {"type": "apiKey", "apiKey": api_key}


def auth_state(result: JsonObject) -> CodexAuth:
    match result:
        case {"account": {"type": "apiKey"}}:
            return CodexAuth.API_KEY
        case {"account": {"type": "chatgpt"}}:
            return CodexAuth.CHATGPT
        case {"account": {"type": str()}}:
            return CodexAuth.OTHER
        case {"requiresOpenaiAuth": False}:
            return CodexAuth.NOT_REQUIRED
        case _:
            return CodexAuth.MISSING


def hub_tools() -> list[JsonObject]:
    return [
        {
            "type": "function",
            "name": HubTool.SEND_FILE.value,
            "description": SEND_FILE_DESCRIPTION,
            "inputSchema": SEND_FILE_SCHEMA,
        },
        {
            "type": "function",
            "name": HubTool.ASK_USER.value,
            "description": ASK_USER_DESCRIPTION,
            "inputSchema": ASK_USER_SCHEMA,
        },
    ]


def open_thread(session: TopicSession, settings: CodexSettings) -> tuple[Call, JsonObject]:
    """`thread/start` for a new topic session, `thread/resume` for a saved one."""
    params: JsonObject = {
        "cwd": str(session.cwd),
        "sandbox": settings.sandbox.value,
        "approvalPolicy": settings.approval.value,
        "approvalsReviewer": APPROVALS_REVIEWER,
    }
    if settings.model is not None:
        params["model"] = settings.model
    if session.session_id is None:
        return Call.THREAD_START, {**params, "dynamicTools": hub_tools()}
    # A resumed thread keeps the dynamic tools it was started with.
    return Call.THREAD_RESUME, {**params, "threadId": session.session_id}


def parse_thread_id(result: JsonObject) -> SessionId:
    match result:
        case {"thread": {"id": str() as thread_id}}:
            return SessionId(thread_id)
        case _:
            raise ProtocolError(f"thread response without an id: {result!r}")


def turn_params(thread_id: SessionId, prompt: Prompt) -> JsonObject:
    return {"threadId": thread_id, "input": user_input(prompt)}


def steer_params(thread_id: SessionId, turn: TurnId, prompt: Prompt) -> JsonObject:
    return {"threadId": thread_id, "expectedTurnId": turn, "input": user_input(prompt)}


def parse_turn_id(result: JsonObject) -> TurnId:
    match result:
        case {"turn": {"id": str() as turn_id}}:
            return TurnId(turn_id)
        case _:
            raise ProtocolError(f"turn response without an id: {result!r}")


def user_input(prompt: Prompt) -> list[JsonObject]:
    images: list[JsonObject] = [
        {"type": "image", "url": _data_url(image)} for image in prompt.images
    ]
    text: list[JsonObject] = [{"type": "text", "text": prompt.text}] if prompt.text.strip() else []
    return [*images, *text]


def _data_url(image: Image) -> str:
    data = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.media_type.value};base64,{data}"


class TurnTracker:
    """Notifications of one thread → hub events; remembers the thread's token use."""

    def __init__(self, thread_id: SessionId) -> None:
        self._thread_id = thread_id
        self._tokens: int | None = None

    def translate(self, notification: Notification) -> Translation:
        try:
            notice = Notice(notification.method)
        except ValueError:
            return Translation()
        params = notification.params
        match notice:
            case Notice.ITEM_STARTED:
                return Translation(_optional(tool_call(params.get("item"))))
            case Notice.ITEM_COMPLETED:
                return Translation(_optional(agent_text(params.get("item"))))
            case Notice.TOKEN_USAGE:
                match params:
                    case {"tokenUsage": {"total": {"totalTokens": int() as total}}}:
                        self._tokens = total
                    case _:
                        pass
                return Translation()
            case Notice.TURN_COMPLETED:
                return self._completed(params.get("turn"))
            case _:
                assert_never(notice)

    def _completed(self, turn: object) -> Translation:
        match turn:
            case {"id": str() as turn_id, "status": "completed"}:
                finished = Finished(self._thread_id, None, None, tokens=self._tokens)
                return Translation((finished,), TurnId(turn_id))
            case {"id": str() as turn_id, "status": "interrupted"}:
                return Translation((Failed("Ход Codex прерван"),), TurnId(turn_id))
            case {"id": str() as turn_id, "error": {"message": str() as message}}:
                return Translation((Failed(message),), TurnId(turn_id))
            case {"id": str() as turn_id}:
                return Translation((Failed("Ход Codex завершился ошибкой"),), TurnId(turn_id))
            case _:
                raise ProtocolError(f"turn/completed without a turn: {turn!r}")


def tool_call(item: object) -> ToolCall | None:
    match item:
        case {"type": "commandExecution", "command": str() as command}:
            return ToolCall(CodexTool.SHELL, truncate(command, TOOL_SUMMARY_LIMIT))
        case {"type": "fileChange", "changes": list() as changes}:
            paths = [
                change["path"]
                for change in changes
                if isinstance(change, dict) and isinstance(change.get("path"), str)
            ]
            summary = ", ".join(paths) or UNNAMED_FILE_CHANGE
            return ToolCall(CodexTool.PATCH, truncate(summary, TOOL_SUMMARY_LIMIT))
        case {
            "type": "mcpToolCall",
            "server": str() as server,
            "tool": str() as name,
            "arguments": raw,
        }:
            arguments = json.dumps(raw, ensure_ascii=False)
            return ToolCall(f"{server}/{name}", truncate(arguments, TOOL_SUMMARY_LIMIT))
        case {"type": "mcpToolCall", "server": str() as server, "tool": str() as name}:
            return ToolCall(f"{server}/{name}", "{}")
        case {"type": "webSearch", "query": str() as query}:
            return ToolCall(CodexTool.WEB_SEARCH, truncate(query, TOOL_SUMMARY_LIMIT))
        case _:
            return None


def agent_text(item: object) -> AssistantText | None:
    match item:
        case {"type": "agentMessage", "text": str() as text} if text.strip():
            return AssistantText(text)
        case _:
            return None


def _optional(event: AgentEvent | None) -> tuple[AgentEvent, ...]:
    return () if event is None else (event,)
