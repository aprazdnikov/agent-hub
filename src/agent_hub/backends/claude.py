"""Claude Code backend via the Claude Agent SDK."""

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Iterator, Mapping
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Literal, Protocol, assert_never

from claude_agent_sdk import (
    TERMINAL_TASK_STATUSES,
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
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

from agent_hub.backends import Inbox, UserChannel
from agent_hub.backends.common import (
    SEND_FILE_DESCRIPTION,
    SEND_FILE_SCHEMA,
    TOOL_SUMMARY_LIMIT,
    HubTool,
    ToolError,
    ToolSuccess,
    deliver_file,
    parse_questions,
)
from agent_hub.config import ClaudeSettings, PermissionMode
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    Answered,
    AssistantText,
    BackgroundAbandoned,
    Denied,
    Failed,
    Finished,
    Prompt,
    SessionId,
    SessionStarted,
    ToolCall,
    ToolRequest,
    TopicSession,
)
from agent_hub.render import truncate

ASK_USER_QUESTION = "AskUserQuestion"
HUB_SERVER = "agent-hub"
SEND_FILE_TOOL = f"mcp__{HUB_SERVER}__{HubTool.SEND_FILE}"
# After a background task reports, the CLI starts a turn of its own within this window.
SETTLE_SECONDS = 30
# Tools whose result the user already sees as its own message, not as a tool line.
_SILENT_TOOLS = frozenset({ASK_USER_QUESTION, SEND_FILE_TOOL})
# Load the operator's own Claude Code setup (CLAUDE.md, permission allowlists, skills,
# MCP servers) so a topic behaves like `claude` started in the same directory.
SETTING_SOURCES: list[Literal["user", "project", "local"]] = ["user", "project", "local"]


class ClaudeBackend:
    def __init__(self, settings: ClaudeSettings, background_timeout_seconds: int) -> None:
        self._settings = settings
        self._background_timeout_seconds = background_timeout_seconds

    async def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel, inbox: Inbox
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
            # Echoes each prompt when a turn takes it in: prompts sent mid-turn are merged
            # into that turn, so counting results cannot tell when all prompts are answered.
            extra_args={"replay-user-messages": None},
        )
        activity = SessionActivity()
        try:
            async with ClaudeSDKClient(options=options) as client:
                activity.sent(await _send(client, prompt))
                async for event in self.converse(client, inbox, activity):
                    yield event
        except ClaudeSDKError as error:
            # The CLI exiting after its last result (e.g. ResultError) is not a failure.
            if activity.awaiting_result:
                yield Failed(f"{type(error).__name__}: {error}")
            return
        if activity.awaiting_result:
            yield Failed("Claude завершился без результата")

    async def converse(
        self, client: "Conversation", inbox: Inbox, activity: "SessionActivity"
    ) -> AsyncGenerator[AgentEvent, None]:
        """Relay messages and further prompts until the session has nothing left to do.

        Closing the client kills the CLI with its background tasks, so it stays open while
        they run; their results arrive as turns the CLI starts on its own.
        """
        tracker = SessionTracker()
        messages = client.receive_messages()
        next_message = asyncio.ensure_future(anext(messages))
        next_prompt = asyncio.ensure_future(inbox.get())
        deadline: float | None = None
        try:
            # A prompt already taken from the inbox must be sent even if the session is idle.
            while (phase := activity.phase) is not Phase.IDLE or next_prompt.done():
                deadline = self._deadline(phase, deadline)
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, _ = await asyncio.wait(
                    {next_message, next_prompt},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    if phase is Phase.BACKGROUND:
                        yield BackgroundAbandoned(activity.background)
                    return
                if next_prompt in done:
                    activity.sent(await _send(client, next_prompt.result()))
                    next_prompt = asyncio.ensure_future(inbox.get())
                if next_message in done:
                    try:
                        message = next_message.result()
                    except StopAsyncIteration:
                        return
                    activity.observe(message)
                    for event in tracker.translate(message, len(activity.background)):
                        yield event
                    next_message = asyncio.ensure_future(anext(messages))
        finally:
            next_message.cancel()
            next_prompt.cancel()

    def _deadline(self, phase: "Phase", current: float | None) -> float | None:
        """Absolute time to give up waiting in `phase`; None waits indefinitely."""
        match phase:
            case Phase.BUSY | Phase.IDLE:
                # A working agent is only stopped by the user (/stop).
                return None
            case Phase.BACKGROUND:
                if current is not None:
                    return current
                return time.monotonic() + self._background_timeout_seconds
            case Phase.SETTLING:
                return time.monotonic() + SETTLE_SECONDS
            case _:
                assert_never(phase)


class Conversation(Protocol):
    """The part of `ClaudeSDKClient` a session needs."""

    def receive_messages(self) -> AsyncIterator[Message]: ...

    async def query(self, prompt: str | AsyncIterable[dict[str, Any]]) -> None: ...


async def _send(client: Conversation, prompt: Prompt) -> str:
    """Send `prompt` under a fresh id, which the CLI echoes back when it takes it in."""
    prompt_id = str(uuid.uuid4())
    await client.query(_one(user_message(prompt, prompt_id)))
    return prompt_id


class Phase(Enum):
    BUSY = auto()  # the agent is working on a turn
    BACKGROUND = auto()  # idle, but background tasks are still running
    SETTLING = auto()  # tasks reported; the CLI is about to start a turn with their results
    IDLE = auto()  # nothing left; the session can be closed


class SessionActivity:
    """Tracks what the CLI session is still doing, to know when closing it loses nothing.

    Only backgrounded tasks matter: foreground subagents finish inside their turn, while a
    background task reports in a turn the CLI starts on its own after the task ends.
    """

    def __init__(self) -> None:
        self._waiting: set[str] = set()  # sent prompts the CLI has not taken in yet
        self._answering = False  # a turn with our prompts is in progress
        self._tasks: dict[str, str] = {}  # running background task id -> description
        self._report_due = False  # a background task ended; its report turn has not begun
        self._reporting = False  # a report turn started by the CLI is in progress

    def sent(self, prompt_id: str) -> None:
        self._waiting.add(prompt_id)

    @property
    def awaiting_result(self) -> bool:
        return bool(self._waiting) or self._answering

    @property
    def background(self) -> tuple[str, ...]:
        return tuple(self._tasks.values())

    @property
    def phase(self) -> Phase:
        if self.awaiting_result or self._reporting:
            return Phase.BUSY
        if self._tasks:
            return Phase.BACKGROUND
        if self._report_due:
            # Also covers a task killed without a report: the settling window closes it.
            return Phase.SETTLING
        return Phase.IDLE

    def observe(self, message: Message) -> None:
        match message:
            case UserMessage(uuid=str(prompt_id)) if prompt_id in self._waiting:
                self._waiting.discard(prompt_id)
                self._answering = True
            case TaskStartedMessage() if message.data.get("is_backgrounded") is not False:
                self._tasks[message.task_id] = message.description
            case TaskNotificationMessage():
                self._task_ended(message.task_id)
            case TaskUpdatedMessage() if _is_terminal(message):
                # Arrives before the notification, so it alone must not close the session.
                self._task_ended(message.task_id)
            case ResultMessage() if _is_injected(message):
                self._reporting = False
            case ResultMessage():
                self._answering = False
            case AssistantMessage() | SystemMessage(subtype="init") if (
                self._report_due and not self._answering
            ):
                self._report_due = False
                self._reporting = True
            case _:
                pass

    def _task_ended(self, task_id: str) -> None:
        if self._tasks.pop(task_id, None) is not None:
            self._report_due = True


def _is_terminal(message: TaskUpdatedMessage) -> bool:
    status = message.status or message.patch.get("status")
    return status in TERMINAL_TASK_STATUSES


def _is_injected(message: ResultMessage) -> bool:
    return message.origin is not None and message.origin["kind"] != "human"


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
    @tool(HubTool.SEND_FILE, SEND_FILE_DESCRIPTION, SEND_FILE_SCHEMA)
    async def send_file(args: dict[str, Any]) -> dict[str, Any]:
        return await send_file_result(args, channel)

    return create_sdk_mcp_server(HUB_SERVER, tools=[send_file])


async def send_file_result(args: Mapping[str, Any], channel: UserChannel) -> dict[str, Any]:
    result = await deliver_file(args, channel)
    match result:
        case ToolSuccess(text):
            return _tool_text(text, is_error=False)
        case ToolError(text):
            return _tool_text(text, is_error=True)
        case _:
            assert_never(result)


def _tool_text(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


def user_message(prompt: Prompt, prompt_id: str) -> dict[str, Any]:
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
        "uuid": prompt_id,
    }


async def _one(message: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    yield message


class SessionTracker:
    """Turns SDK messages into domain events, remembering session id and termination."""

    def __init__(self) -> None:
        self.session_id: SessionId | None = None
        self.terminated = False

    def translate(self, message: Message, background: int = 0) -> Iterator[AgentEvent]:
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
                    background=background,
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
