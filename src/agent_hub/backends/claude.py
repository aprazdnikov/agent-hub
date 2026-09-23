"""Claude Code backend via the Claude Agent SDK."""

import json
from collections.abc import AsyncGenerator, Iterator, Mapping
from decimal import Decimal
from typing import Any, Literal, assert_never

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
)

from agent_hub.backends import ApprovalGate
from agent_hub.config import ClaudeSettings, PermissionMode
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    AssistantText,
    Denied,
    Failed,
    Finished,
    SessionId,
    SessionStarted,
    ToolCall,
    ToolRequest,
    TopicSession,
)
from agent_hub.render import truncate

TOOL_SUMMARY_LIMIT = 600
# Load the operator's own Claude Code setup (CLAUDE.md, permission allowlists, skills,
# MCP servers) so a topic behaves like `claude` started in the same directory.
SETTING_SOURCES: list[Literal["user", "project", "local"]] = ["user", "project", "local"]


class ClaudeBackend:
    def __init__(self, settings: ClaudeSettings) -> None:
        self._settings = settings

    async def run(
        self, session: TopicSession, prompt: str, gate: ApprovalGate
    ) -> AsyncGenerator[AgentEvent, None]:
        async def can_use_tool(
            tool_name: str, tool_input: dict[str, Any], _context: ToolPermissionContext
        ) -> PermissionResultAllow | PermissionResultDeny:
            request = ToolRequest(tool_name, summarize_tool_input(tool_name, tool_input))
            decision = await gate.request(request)
            match decision:
                case Allowed():
                    return PermissionResultAllow()
                case Denied(reason):
                    return PermissionResultDeny(message=reason)
                case _:
                    assert_never(decision)

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
        )
        tracker = SessionTracker()
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
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
                elif isinstance(block, ToolUseBlock):
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
