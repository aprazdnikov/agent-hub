"""Claude Code backend via the Claude Agent SDK."""

import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
from decimal import Decimal
from typing import Any, Literal, assert_never

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    McpSdkServerConfig,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from agent_hub.backends import UserChannel
from agent_hub.config import ClaudeSettings, PermissionMode
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    Answered,
    AssistantText,
    Delivered,
    Denied,
    Failed,
    Finished,
    OutgoingFile,
    Prompt,
    Question,
    QuestionOption,
    SessionId,
    SessionStarted,
    ToolCall,
    ToolRequest,
    TopicSession,
)
from agent_hub.render import truncate

TOOL_SUMMARY_LIMIT = 600
ASK_USER_QUESTION = "AskUserQuestion"
HUB_SERVER = "agent-hub"
SEND_FILE = "send_file"
SEND_FILE_TOOL = f"mcp__{HUB_SERVER}__{SEND_FILE}"
_SEND_FILE_DESCRIPTION = (
    "Send a file to the user in their Telegram chat. The user only sees your text replies, "
    "so use this whenever they ask for a file or a file is the natural result (a PDF report, "
    "an archive, an image, a CSV export). `path` is absolute or relative to the working "
    "directory and must stay inside it; `caption` is optional text shown under the file."
)
_SEND_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "caption": {"type": "string"}},
    "required": ["path"],
}
# Tools whose result the user already sees as its own message, not as a tool line.
_SILENT_TOOLS = frozenset({ASK_USER_QUESTION, SEND_FILE_TOOL})
# Load the operator's own Claude Code setup (CLAUDE.md, permission allowlists, skills,
# MCP servers) so a topic behaves like `claude` started in the same directory.
SETTING_SOURCES: list[Literal["user", "project", "local"]] = ["user", "project", "local"]


class ClaudeBackend:
    def __init__(self, settings: ClaudeSettings) -> None:
        self._settings = settings

    async def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel
    ) -> AsyncGenerator[AgentEvent, None]:
        async def can_use_tool(
            tool_name: str, tool_input: dict[str, Any], _context: ToolPermissionContext
        ) -> PermissionResultAllow | PermissionResultDeny:
            return await decide(tool_name, tool_input, channel)

        options = ClaudeAgentOptions(
            cwd=session.cwd,
            resume=session.session_id,
            permission_mode=_sdk_permission_mode(self._settings.permission_mode),
            model=self._settings.model,
            max_budget_usd=(
                None
                if self._settings.max_budget_usd is None
                else float(self._settings.max_budget_usd)
            ),
            setting_sources=SETTING_SOURCES,
            can_use_tool=can_use_tool,
            mcp_servers={HUB_SERVER: _hub_server(channel)},
        )
        tracker = SessionTracker()
        try:
            async with ClaudeSDKClient(options=options) as client:
                if prompt.images:
                    await client.query(_one(user_message(prompt)))
                else:
                    await client.query(prompt.text)
                async for message in client.receive_response():
                    for event in tracker.translate(message):
                        yield event
        except ClaudeSDKError as error:
            # ResultError is raised after its ResultMessage was already translated.
            if not tracker.terminated:
                yield Failed(f"{type(error).__name__}: {error}")
            return
        if not tracker.terminated:
            yield Failed("Claude завершился без результата")


async def decide(
    tool_name: str, tool_input: dict[str, Any], channel: UserChannel
) -> PermissionResultAllow | PermissionResultDeny:
    """Permission callback: clarifying questions go to the human as questions, the rest as
    approvals; a question input we cannot parse still reaches the human as an approval."""
    if tool_name == SEND_FILE_TOOL:
        # Only reaches the user's own chat, which already sees everything the agent prints.
        return PermissionResultAllow()
    questions = parse_questions(tool_input) if tool_name == ASK_USER_QUESTION else None
    if questions is not None:
        outcome = await channel.ask(questions)
        match outcome:
            case Answered(answers):
                # The CLI reads the answers from the tool input and hands them to the model.
                return PermissionResultAllow(updated_input={**tool_input, "answers": dict(answers)})
            case Denied(reason):
                return PermissionResultDeny(message=reason)
            case _:
                assert_never(outcome)
    decision = await channel.request(
        ToolRequest(tool_name, summarize_tool_input(tool_name, tool_input))
    )
    match decision:
        case Allowed():
            return PermissionResultAllow()
        case Denied(reason):
            return PermissionResultDeny(message=reason)
        case _:
            assert_never(decision)


def _hub_server(channel: UserChannel) -> McpSdkServerConfig:
    @tool(SEND_FILE, _SEND_FILE_DESCRIPTION, _SEND_FILE_SCHEMA)
    async def send_file(args: dict[str, Any]) -> dict[str, Any]:
        return await send_file_result(args, channel)

    return create_sdk_mcp_server(HUB_SERVER, tools=[send_file])


async def send_file_result(args: Mapping[str, Any], channel: UserChannel) -> dict[str, Any]:
    """`send_file` tool body: a failed delivery is a tool error the agent can react to."""
    path, caption = args.get("path"), args.get("caption", "")
    if not isinstance(path, str) or not path.strip() or not isinstance(caption, str):
        return _tool_text("path must be a non-empty string, caption a string", is_error=True)
    delivery = await channel.send_file(OutgoingFile(path, caption))
    match delivery:
        case Delivered():
            return _tool_text(f"Файл {path} отправлен пользователю", is_error=False)
        case Denied(reason):
            return _tool_text(reason, is_error=True)
        case _:
            assert_never(delivery)


def _tool_text(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def parse_questions(tool_input: Mapping[str, Any]) -> tuple[Question, ...] | None:
    """`AskUserQuestion` input → questions; None when it does not match the expected shape."""
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
        option = _parse_option(item)
        if option is None:
            return None
        parsed.append(option)
    return Question(text, header, tuple(parsed), multi_select=raw.get("multiSelect") is True)


def _parse_option(raw: object) -> QuestionOption | None:
    if not isinstance(raw, dict):
        return None
    label, description = raw.get("label"), raw.get("description", "")
    if not isinstance(label, str) or not label.strip() or not isinstance(description, str):
        return None
    return QuestionOption(label, description)


def user_message(prompt: Prompt) -> dict[str, Any]:
    """Stream-json user message with images first, as the Messages API recommends."""
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type.value,
                "data": base64.b64encode(image.data).decode("ascii"),
            },
        }
        for image in prompt.images
    ]
    if prompt.text.strip():
        content.append({"type": "text", "text": prompt.text})
    return {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


async def _one(message: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    yield message


class SessionTracker:
    """Turns SDK messages into domain events, remembering session id and termination."""

    def __init__(self) -> None:
        self.session_id: SessionId | None = None
        self.terminated = False

    def translate(self, message: Message) -> Iterator[AgentEvent]:
        found = _session_id_of(message)
        if found is not None and found != self.session_id:
            self.session_id = found
            yield SessionStarted(found)
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    yield AssistantText(block.text)
                elif isinstance(block, ToolUseBlock) and block.name not in _SILENT_TOOLS:
                    yield ToolCall(block.name, summarize_tool_input(block.name, block.input))
        elif isinstance(message, ResultMessage):
            self.terminated = True
            if message.is_error:
                yield Failed(f"{message.subtype}: {message.result or 'ошибка выполнения'}")
            else:
                yield Finished(
                    session_id=SessionId(message.session_id),
                    turns=message.num_turns,
                    cost_usd=(
                        None
                        if message.total_cost_usd is None
                        else Decimal(str(message.total_cost_usd))
                    ),
                )


def _session_id_of(message: Message) -> SessionId | None:
    if isinstance(message, SystemMessage):
        raw = message.data.get("session_id")
        return SessionId(raw) if isinstance(raw, str) else None
    if isinstance(message, AssistantMessage | ResultMessage) and message.session_id:
        return SessionId(message.session_id)
    return None


def summarize_tool_input(tool: str, tool_input: Mapping[str, Any]) -> str:
    """One human-readable line for the most common Claude Code tools."""
    key = {
        "Bash": "command",
        "Read": "file_path",
        "Write": "file_path",
        "Edit": "file_path",
        "MultiEdit": "file_path",
        "NotebookEdit": "notebook_path",
        "Glob": "pattern",
        "Grep": "pattern",
        "WebFetch": "url",
        "WebSearch": "query",
    }.get(tool)
    value = tool_input.get(key) if key is not None else None
    text = value if isinstance(value, str) else json.dumps(tool_input, ensure_ascii=False)
    return truncate(text, TOOL_SUMMARY_LIMIT)


def _sdk_permission_mode(
    mode: PermissionMode,
) -> Literal["default", "acceptEdits", "plan", "bypassPermissions"]:
    match mode:
        case PermissionMode.DEFAULT:
            return "default"
        case PermissionMode.ACCEPT_EDITS:
            return "acceptEdits"
        case PermissionMode.PLAN:
            return "plan"
        case PermissionMode.BYPASS_PERMISSIONS:
            return "bypassPermissions"
        case _:
            assert_never(mode)
