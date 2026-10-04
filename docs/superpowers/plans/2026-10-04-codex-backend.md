# Codex Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI Codex as a second per-topic agent next to Claude, with API-key or ChatGPT-subscription auth and feature parity (text, tool lines, approvals, files, images, steering, questions).

**Architecture:** A small asyncio JSON-RPC client (`backends/rpc.py`) talks to `codex app-server` over stdio. Pure protocol mapping lives in `codex_protocol.py` (requests, responses, notifications → hub events) and `codex_requests.py` (server requests → `UserChannel`). `codex.py` owns the process, the session loop and the startup auth probe. Shared hub tools move from `claude.py` to `common.py`.

**Tech Stack:** Python 3.12, asyncio, `openai-codex-cli-bin` (bundled `codex` binary, 0.160.0), python-telegram-bot 22, pytest + pytest-asyncio, ruff (`select = ["ALL"]`), mypy strict, uv.

**Spec:** `docs/superpowers/specs/2026-10-04-codex-backend-design.md`

## Global Constraints

- Python 3.12 syntax (PEP 695 generics allowed); `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest` must pass after every task.
- No comments except *why*; docstrings only where non-obvious; code and docstrings in English, user-facing strings in Russian.
- Frozen `slots` dataclasses with `@final`; enums instead of boolean flags; exhaustive `match` with `assert_never` on own types.
- Child processes inherit the hub environment (same as Claude today).
- Codex protocol facts are fixed by the spec's "Протокол" section (codex-cli 0.160.0); do not invent other methods or fields.
- Commit messages in Russian, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feature/codex-backend`.

## Review Focus

- Codex binary missing or not runnable (wrong wheel, broken install) → the topic gets `Failed("Codex app-server завершился…")`, the hub keeps serving other topics. Test: Task 6 `test_spawn_failure_fails_the_turn`.
- Topic switched to Codex without any Codex login → one clear `Failed` naming `codex login` / `OPENAI_API_KEY`, no hang. Test: Task 6 `test_missing_login_fails_with_a_hint` (real binary, empty `CODEX_HOME`).
- Saved Codex thread no longer exists (volume wiped, thread deleted) → `Failed` with the `/reset` hint, no silent new thread. Test: Task 6 `test_resume_failure_suggests_reset`.
- The user taps nothing for minutes on an approval while Codex emits other traffic → transport keeps flowing. Test: Task 3 `test_pending_server_request_does_not_block_responses`.
- A message sent just as a turn ends → steer is rejected, a new turn starts, and the old turn's completion does not close the session. Test: Task 6 `test_rejected_steer_starts_a_new_turn`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/agent_hub/backends/common.py` (new) | Hub tools (`HubTool`, descriptions, schemas), `ToolResult`, `deliver_file`, `parse_questions`, `parse_option`, `TOOL_SUMMARY_LIMIT` |
| `src/agent_hub/backends/rpc.py` (new) | `RpcConnection`, `Connection` protocol, errors, `Notification` |
| `src/agent_hub/backends/codex_protocol.py` (new) | `Call`, `CodexTool`, `CodexAuth`, `TurnId`, request params, response parsing, `TurnTracker` |
| `src/agent_hub/backends/codex_requests.py` (new) | `answer(channel, method, params)` for app-server requests |
| `src/agent_hub/backends/codex.py` (new) | `app_server`, `CodexBackend`, `converse`, `probe_auth` |
| `src/agent_hub/backends/claude.py` | Uses `common.py`; takes `background_timeout_seconds` explicitly |
| `src/agent_hub/domain.py` | `BackendKind.CODEX`; `Finished.turns` optional, `Finished.tokens` |
| `src/agent_hub/config.py` | `Settings.default_backend`, `Settings.background_timeout_seconds`, `CodexSettings` |
| `src/agent_hub/render.py` | `format_finished` with optional parts and tokens |
| `src/agent_hub/commands.py` | `parse_new_args(args, default)`, `parse_backend_args` |
| `src/agent_hub/bot.py` | `/backend`, settings-driven default backend and timeout, tokens |
| `src/agent_hub/__main__.py` | Codex wiring, `check_codex` |
| `tests/fakes.py` (new) | `FakeChannel`, `ASK_INPUT`, `ASK_QUESTION` |
| `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml`, `.env.example`, `README.md`, `pyproject.toml` | Packaging and docs |

---

### Task 1: Shared hub tools in `backends/common.py`

**Files:**
- Create: `src/agent_hub/backends/common.py`, `tests/fakes.py`, `tests/test_common.py`
- Modify: `src/agent_hub/backends/claude.py`, `tests/test_claude_backend.py`

**Interfaces:**
- Produces: `TOOL_SUMMARY_LIMIT: int`; `class HubTool(StrEnum)` with `SEND_FILE = "send_file"`, `ASK_USER = "ask_user"`; `SEND_FILE_DESCRIPTION: str`, `SEND_FILE_SCHEMA: dict[str, Any]`, `ASK_USER_DESCRIPTION: str`, `ASK_USER_SCHEMA: dict[str, Any]`; `ToolSuccess(text: str)`, `ToolError(text: str)`, `ToolResult = ToolSuccess | ToolError`; `async def deliver_file(args: Mapping[str, Any], channel: UserChannel) -> ToolResult`; `def parse_questions(tool_input: Mapping[str, Any]) -> tuple[Question, ...] | None`; `def parse_option(raw: object) -> QuestionOption | None`.
- Produces (tests): `tests.fakes.FakeChannel(decision, outcome, delivery=Delivered())` with `.requests`, `.asked`, `.sent`; `ASK_INPUT`, `ASK_QUESTION`.

- [ ] **Step 1: Move the fake channel into `tests/fakes.py`**

```python
from collections.abc import Sequence
from typing import Any

from agent_hub.domain import (
    Decision,
    Delivered,
    FileDelivery,
    OutgoingFile,
    Question,
    QuestionOption,
    QuestionsOutcome,
    ToolRequest,
)

ASK_INPUT: dict[str, Any] = {
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
ASK_QUESTION = Question(
    "Какой формат?",
    "Формат",
    (QuestionOption("Кратко", "Только суть"), QuestionOption("Подробно", "С примерами")),
    multi_select=False,
)


class FakeChannel:
    def __init__(
        self,
        decision: Decision,
        outcome: QuestionsOutcome,
        delivery: FileDelivery = Delivered(),
    ) -> None:
        self.decision = decision
        self.outcome = outcome
        self.delivery = delivery
        self.requests: list[ToolRequest] = []
        self.asked: list[Sequence[Question]] = []
        self.sent: list[OutgoingFile] = []

    async def send_file(self, file: OutgoingFile) -> FileDelivery:
        self.sent.append(file)
        return self.delivery

    async def request(self, tool: ToolRequest) -> Decision:
        self.requests.append(tool)
        return self.decision

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome:
        self.asked.append(questions)
        return self.outcome
```

In `tests/test_claude_backend.py`: delete `_ASK_INPUT`, `_ASK_QUESTION`, `DELIVERED`, the `FakeChannel` class, `test_questions_are_parsed` and `test_malformed_questions_are_rejected`; add `from tests.fakes import ASK_INPUT, ASK_QUESTION, FakeChannel`; rename uses `_ASK_INPUT` → `ASK_INPUT`, `_ASK_QUESTION` → `ASK_QUESTION`; drop `parse_questions` from the `agent_hub.backends.claude` import. (If ruff B008 flags `Delivered()` as a default argument, use a module constant `DELIVERED = Delivered()` in `fakes.py` as the default.)

- [ ] **Step 2: Write the failing tests `tests/test_common.py`**

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_common.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_hub.backends.common'`.

- [ ] **Step 4: Create `src/agent_hub/backends/common.py`**

```python
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
```

- [ ] **Step 5: Switch `claude.py` to `common.py`**

In `src/agent_hub/backends/claude.py`:
1. Delete `TOOL_SUMMARY_LIMIT = 600`, `SEND_FILE = "send_file"`, `_SEND_FILE_DESCRIPTION`, `_SEND_FILE_SCHEMA`, and the functions `parse_questions`, `_parse_question`, `_parse_option`.
2. Add:
```python
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
```
3. `SEND_FILE_TOOL = f"mcp__{HUB_SERVER}__{HubTool.SEND_FILE}"`.
4. In `_hub_server`: `@tool(HubTool.SEND_FILE, SEND_FILE_DESCRIPTION, SEND_FILE_SCHEMA)`.
5. Replace the body of `send_file_result`:
```python
async def send_file_result(args: Mapping[str, Any], channel: UserChannel) -> dict[str, Any]:
    result = await deliver_file(args, channel)
    match result:
        case ToolSuccess(text):
            return _tool_text(text, is_error=False)
        case ToolError(text):
            return _tool_text(text, is_error=True)
        case _:
            assert_never(result)
```
6. Run `uv run ruff check --fix src tests` to drop now-unused imports (`Delivered`, `OutgoingFile`, `Question`, `QuestionOption` in `claude.py`).

- [ ] **Step 6: Run all checks**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q`
Expected: all pass; `tests/test_claude_backend.py` unchanged in behaviour.

- [ ] **Step 7: Commit**

```bash
git add src/agent_hub/backends tests
git commit -m "Общие инструменты хаба вынесены из бэкенда Claude

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Backend-neutral settings, `Finished` and its rendering

**Files:**
- Modify: `src/agent_hub/domain.py`, `src/agent_hub/render.py`, `src/agent_hub/config.py`, `src/agent_hub/commands.py`, `src/agent_hub/backends/claude.py`, `src/agent_hub/bot.py`, `src/agent_hub/__main__.py`
- Test: `tests/test_render.py`, `tests/test_config.py`, `tests/test_commands.py`, `tests/test_activity.py`

**Interfaces:**
- Produces: `Finished(session_id, turns: int | None, cost_usd, background=0, tokens: int | None = None)`; `format_finished(turns: int | None, cost_usd: Decimal | None, background: int = 0, tokens: int | None = None) -> str`; `CodexSandbox`, `CodexApproval` (StrEnum); `CodexSettings(model: str | None, sandbox: CodexSandbox, approval: CodexApproval, api_key: str | None)`; `Settings.background_timeout_seconds: int`, `Settings.default_backend: BackendKind`, `Settings.codex: CodexSettings`; `ClaudeSettings(permission_mode, model, max_budget_usd)`; `ClaudeBackend(settings: ClaudeSettings, background_timeout_seconds: int)`; `parse_new_args(args: Sequence[str], default: BackendKind) -> NewSessionArgs`.

- [ ] **Step 1: Failing render tests** — append to `tests/test_render.py`:

```python
def test_format_finished_without_turns_shows_tokens() -> None:
    assert format_finished(None, None, tokens=12345) == "✅ Готово · токенов в сессии: 12 345"


def test_format_finished_with_nothing_measured() -> None:
    assert format_finished(None, None) == "✅ Готово"
```

- [ ] **Step 2: Failing config tests** — in `tests/test_config.py` import `CodexApproval, CodexSandbox` and `BackendKind` (from `agent_hub.domain`); in `test_minimal_env_uses_defaults` replace the last assert with:

```python
    assert settings.background_timeout_seconds == DEFAULT_BACKGROUND_TIMEOUT_SECONDS
    assert settings.default_backend is BackendKind.CLAUDE
    assert settings.codex.model is None
    assert settings.codex.sandbox is CodexSandbox.WORKSPACE_WRITE
    assert settings.codex.approval is CodexApproval.ON_REQUEST
    assert settings.codex.api_key is None
```

In `test_optional_values_are_parsed` replace `settings.claude.background_timeout_seconds` with `settings.background_timeout_seconds`, and add:

```python
def test_codex_values_are_parsed(env: dict[str, str]) -> None:
    env |= {
        "AGENT_HUB_CODEX_MODEL": "gpt-6.1-sol",
        "AGENT_HUB_CODEX_SANDBOX": "danger-full-access",
        "AGENT_HUB_CODEX_APPROVAL": "never",
        "OPENAI_API_KEY": " sk-test ",
    }
    settings = load_settings(env)

    assert settings.codex.model == "gpt-6.1-sol"
    assert settings.codex.sandbox is CodexSandbox.DANGER_FULL_ACCESS
    assert settings.codex.approval is CodexApproval.NEVER
    assert settings.codex.api_key == "sk-test"
    assert "sk-test" not in repr(settings)
```

Add to the `test_invalid_values_are_rejected` parameters:

```python
        ("AGENT_HUB_DEFAULT_BACKEND", "gemini"),
        ("AGENT_HUB_CODEX_SANDBOX", "none"),
        ("AGENT_HUB_CODEX_APPROVAL", "always"),
```

- [ ] **Step 3: Failing commands test** — in `tests/test_commands.py` change the call to `parse_new_args(args, BackendKind.CLAUDE)`.

- [ ] **Step 4: Run to verify failure**

Run: `uv run pytest tests/test_render.py tests/test_config.py tests/test_commands.py -q`
Expected: FAIL (`format_finished` rejects `None` output, unknown attributes, unexpected argument).

- [ ] **Step 5: Implement `domain.py` and `render.py`**

`domain.py` — replace `Finished` and the `ImageMediaType` docstring:

```python
class ImageMediaType(StrEnum):
    """Image formats every backend accepts inline."""
```

```python
@final
@dataclass(frozen=True, slots=True)
class Finished:
    """One agent turn ended; `background` tasks keep running and report in later turns.

    Each backend reports what its agent measures: Claude turns and cost, Codex tokens.
    """

    session_id: SessionId
    turns: int | None
    cost_usd: Decimal | None
    background: int = 0
    tokens: int | None = None
```

`render.py`:

```python
def format_finished(
    turns: int | None, cost_usd: Decimal | None, background: int = 0, tokens: int | None = None
) -> str:
    parts = ["✅ Готово"]
    if turns is not None:
        parts.append(f"ходов: {turns}")
    if cost_usd is not None:
        parts.append(f"${cost_usd.quantize(Decimal('0.01'))}")
    if tokens is not None:
        parts.append(f"токенов в сессии: {tokens:_}".replace("_", " "))
    if background:
        parts.append(f"⏳ в фоне задач: {background}, пришлю результат")
    return " · ".join(parts)
```

- [ ] **Step 6: Implement `config.py`**

Add `from dataclasses import dataclass, field` and `from agent_hub.domain import BackendKind`. Add after `PREFIX`:

```python
# Codex's own variable name, so an existing key works without renaming.
OPENAI_API_KEY = "OPENAI_API_KEY"
```

Add after `PermissionMode`:

```python
class CodexSandbox(StrEnum):
    """What the OS lets commands started by Codex touch."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class CodexApproval(StrEnum):
    """When Codex asks the human before acting."""

    UNTRUSTED = "untrusted"
    ON_REQUEST = "on-request"
    NEVER = "never"
```

Replace `ClaudeSettings` and `Settings`, add `CodexSettings`:

```python
@final
@dataclass(frozen=True, slots=True)
class ClaudeSettings:
    permission_mode: PermissionMode
    model: str | None
    max_budget_usd: Decimal | None


@final
@dataclass(frozen=True, slots=True)
class CodexSettings:
    model: str | None
    sandbox: CodexSandbox
    approval: CodexApproval
    api_key: str | None = field(repr=False)


@final
@dataclass(frozen=True, slots=True)
class Settings:
    telegram_token: str
    chat_id: int
    allowed_user_ids: frozenset[int]
    workspace_root: Path
    state_file: Path
    approval_timeout_seconds: int
    background_timeout_seconds: int
    default_backend: BackendKind
    claude: ClaudeSettings
    codex: CodexSettings
```

In `load_settings`, after `approval_timeout_seconds=...`:

```python
        background_timeout_seconds=_parse_positive_int(
            _optional(env, "BACKGROUND_TIMEOUT_SECONDS"),
            "BACKGROUND_TIMEOUT_SECONDS",
            DEFAULT_BACKGROUND_TIMEOUT_SECONDS,
        ),
        default_backend=_choice(env, "DEFAULT_BACKEND", BackendKind.CLAUDE),
        claude=ClaudeSettings(
            permission_mode=_choice(env, "CLAUDE_PERMISSION_MODE", PermissionMode.DEFAULT),
            model=_optional(env, "CLAUDE_MODEL"),
            max_budget_usd=_parse_budget(_optional(env, "CLAUDE_MAX_BUDGET_USD")),
        ),
        codex=CodexSettings(
            model=_optional(env, "CODEX_MODEL"),
            sandbox=_choice(env, "CODEX_SANDBOX", CodexSandbox.WORKSPACE_WRITE),
            approval=_choice(env, "CODEX_APPROVAL", CodexApproval.ON_REQUEST),
            api_key=env.get(OPENAI_API_KEY, "").strip() or None,
        ),
```

Replace `_parse_permission_mode` with:

```python
def _choice[E: StrEnum](env: Mapping[str, str], name: str, default: E) -> E:
    raw = _optional(env, name)
    if raw is None:
        return default
    kind = type(default)
    try:
        return kind(raw)
    except ValueError as error:
        allowed = ", ".join(member.value for member in kind)
        raise ConfigError(f"{PREFIX}{name} must be one of: {allowed}; got {raw!r}") from error
```

- [ ] **Step 7: Implement `commands.py`** — delete `DEFAULT_BACKEND`, change:

```python
def parse_new_args(args: Sequence[str], default: BackendKind) -> NewSessionArgs:
    """`/new [backend] [path]`; a first word that is not a backend name starts the path."""
    if not args:
        return NewSessionArgs(default, None)
    try:
        backend = BackendKind(args[0].lower())
    except ValueError:
        return NewSessionArgs(default, " ".join(args))
    rest = " ".join(args[1:])
    return NewSessionArgs(backend, rest or None)
```

- [ ] **Step 8: Update callers**

`backends/claude.py`:
```python
class ClaudeBackend:
    def __init__(self, settings: ClaudeSettings, background_timeout_seconds: int) -> None:
        self._settings = settings
        self._background_timeout_seconds = background_timeout_seconds
```
and in `_deadline` use `self._background_timeout_seconds`.

`tests/test_activity.py` `_backend`:
```python
def _backend(background_timeout: int = 60) -> ClaudeBackend:
    return ClaudeBackend(ClaudeSettings(PermissionMode.DEFAULT, None, None), background_timeout)
```

`__main__.py`: `return ClaudeBackend(settings.claude, settings.background_timeout_seconds)`.

`bot.py`:
- import `from agent_hub.commands import join_path_args, parse_new_args` (no `DEFAULT_BACKEND`);
- both `TopicSession(DEFAULT_BACKEND, ...)` → `TopicSession(self._settings.default_backend, ...)`;
- `parse_new_args(context.args or [], self._settings.default_backend)`;
- `timeout = self._settings.background_timeout_seconds`;
- event match:
```python
            case Finished(session_id, turns, cost_usd, background, tokens):
                self._store.put(key, self._session(key).with_session(session_id))
                log.info(
                    "turn finished",
                    extra={**_fields(key), "turns": turns, "background": background},
                )
                await sender.text(key, format_finished(turns, cost_usd, background, tokens))
```
- `/cwd` comment: `# Agents tie a session to its directory, so a new cwd needs a new session.`

- [ ] **Step 9: Run all checks**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src tests
git commit -m "Общие настройки бэкендов: бэкенд по умолчанию, таймаут фона, настройки Codex

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: JSON-RPC connection `backends/rpc.py`

**Files:**
- Create: `src/agent_hub/backends/rpc.py`, `tests/test_rpc.py`

**Interfaces:**
- Produces: `JsonObject = dict[str, Any]`; `METHOD_NOT_FOUND = -32601`, `INVALID_PARAMS = -32602`, `INTERNAL_ERROR = -32603`; exceptions `RpcError(code: int, message: str)` (attrs `.code`, `.message`), `TransportClosedError`, `ProtocolError`, `UnsupportedRequestError`, `MalformedRequestError`; `Notification(method: str, params: JsonObject)`; `RequestHandler = Callable[[str, JsonObject], Awaitable[JsonObject]]`; protocols `LineReader`, `LineWriter`, `Connection` (`request(method, params) -> JsonObject`, `notify(method) -> None`, `notification() -> Notification`); `RpcConnection(reader, writer, handler)` implementing `Connection` plus `serve() -> None`.

- [ ] **Step 1: Write failing tests `tests/test_rpc.py`**

```python
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from agent_hub.backends.rpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonObject,
    MalformedRequestError,
    Notification,
    RequestHandler,
    RpcConnection,
    RpcError,
    TransportClosedError,
    UnsupportedRequestError,
)


class FakeWriter:
    def __init__(self) -> None:
        self.sent: asyncio.Queue[JsonObject] = asyncio.Queue()

    def write(self, data: bytes) -> None:
        self.sent.put_nowait(json.loads(data))

    async def drain(self) -> None:
        return None


async def _echo(method: str, params: JsonObject) -> JsonObject:
    match method:
        case "echo":
            return params
        case "malformed":
            raise MalformedRequestError("bad params")
        case _:
            raise UnsupportedRequestError(method)


class Peer:
    """The app-server end of a connection under test."""

    def __init__(self, handler: RequestHandler) -> None:
        self.reader = asyncio.StreamReader()
        self.writer = FakeWriter()
        self.connection = RpcConnection(self.reader, self.writer, handler)

    def send(self, message: JsonObject) -> None:
        self.reader.feed_data(json.dumps(message).encode() + b"\n")

    async def received(self) -> JsonObject:
        return await asyncio.wait_for(self.writer.sent.get(), 1)


@asynccontextmanager
async def serving(handler: RequestHandler = _echo) -> AsyncIterator[Peer]:
    peer = Peer(handler)
    serve = asyncio.create_task(peer.connection.serve())
    try:
        yield peer
    finally:
        peer.reader.feed_eof()
        await asyncio.wait_for(serve, 1)


async def test_request_returns_the_matching_result() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("thread/start", {"cwd": "w"}))

        assert await peer.received() == {"id": 1, "method": "thread/start", "params": {"cwd": "w"}}
        peer.send({"id": 1, "result": {"thread": {"id": "t1"}}})

        assert await asyncio.wait_for(call, 1) == {"thread": {"id": "t1"}}


async def test_error_response_raises() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("thread/resume", {}))
        await peer.received()
        peer.send({"id": 1, "error": {"code": -32600, "message": "no rollout"}})

        with pytest.raises(RpcError) as raised:
            await asyncio.wait_for(call, 1)
        assert (raised.value.code, raised.value.message) == (-32600, "no rollout")


async def test_notifications_arrive_in_order() -> None:
    async with serving() as peer:
        peer.reader.feed_data(b"not json\n")
        peer.send({"method": "turn/started", "params": {"turn": {"id": "a"}}})
        peer.send({"method": "turn/completed"})

        assert await peer.connection.notification() == Notification(
            "turn/started", {"turn": {"id": "a"}}
        )
        assert await peer.connection.notification() == Notification("turn/completed", {})


async def test_notify_has_no_id() -> None:
    async with serving() as peer:
        await peer.connection.notify("initialized")
        assert await peer.received() == {"method": "initialized"}


async def test_server_request_is_answered_by_the_handler() -> None:
    async with serving() as peer:
        peer.send({"id": 7, "method": "echo", "params": {"a": 1}})
        assert await peer.received() == {"id": 7, "result": {"a": 1}}


@pytest.mark.parametrize(
    ("method", "code"), [("unknown/method", METHOD_NOT_FOUND), ("malformed", INVALID_PARAMS)]
)
async def test_refused_server_request_gets_an_error(method: str, code: int) -> None:
    async with serving() as peer:
        peer.send({"id": 8, "method": method, "params": {}})
        reply = await peer.received()
        assert reply["id"] == 8
        assert reply["error"]["code"] == code


async def test_pending_server_request_does_not_block_responses() -> None:
    release = asyncio.Event()

    async def slow(method: str, _params: JsonObject) -> JsonObject:
        await release.wait()
        return {"method": method}

    async with serving(slow) as peer:
        peer.send({"id": 1, "method": "approve", "params": {}})
        call = asyncio.create_task(peer.connection.request("turn/steer", {}))
        request = await peer.received()
        peer.send({"id": request["id"], "result": {"turnId": "t"}})

        assert await asyncio.wait_for(call, 1) == {"turnId": "t"}
        release.set()
        assert await peer.received() == {"id": 1, "result": {"method": "approve"}}


async def test_closed_transport_fails_pending_and_later_calls() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("turn/start", {}))
        await peer.received()
        peer.reader.feed_eof()

        with pytest.raises(TransportClosedError):
            await asyncio.wait_for(call, 1)
        with pytest.raises(TransportClosedError):
            await peer.connection.notification()
        with pytest.raises(TransportClosedError):
            await peer.connection.notification()
        with pytest.raises(TransportClosedError):
            await peer.connection.request("turn/start", {})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_rpc.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_hub.backends.rpc'`.

- [ ] **Step 3: Implement `src/agent_hub/backends/rpc.py`**

```python
"""Newline-delimited JSON-RPC 2.0 over a child process's stdio, in both directions."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, final

log = logging.getLogger(__name__)

JsonObject = dict[str, Any]
RequestId = int | str

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """The peer answered a request with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TransportClosedError(Exception):
    """The peer closed its output; nothing more will arrive."""


class ProtocolError(Exception):
    """The peer sent a message of an unexpected shape."""


class UnsupportedRequestError(Exception):
    """A server request this client does not implement."""


class MalformedRequestError(Exception):
    """A server request whose params do not have the expected shape."""


@final
@dataclass(frozen=True, slots=True)
class Notification:
    method: str
    params: JsonObject


RequestHandler = Callable[[str, JsonObject], Awaitable[JsonObject]]


class LineReader(Protocol):
    async def readline(self) -> bytes: ...


class LineWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class Connection(Protocol):
    """What a session needs from a JSON-RPC peer."""

    async def request(self, method: str, params: JsonObject) -> JsonObject: ...

    async def notify(self, method: str) -> None: ...

    async def notification(self) -> Notification: ...


class RpcConnection:
    """Matches responses to requests, queues notifications and answers server requests.

    Each server request runs in its own task, so one awaiting a human does not stall the
    stream. Progress requires `serve`; once it returns, every call raises
    `TransportClosedError`.
    """

    def __init__(self, reader: LineReader, writer: LineWriter, handler: RequestHandler) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._last_id = 0
        self._pending: dict[RequestId, asyncio.Future[JsonObject]] = {}
        self._notifications: asyncio.Queue[Notification | None] = asyncio.Queue()
        self._answering: set[asyncio.Task[None]] = set()
        self._closed = False

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        self._last_id += 1
        request_id = self._last_id
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str) -> None:
        await self._send({"method": method})

    async def notification(self) -> Notification:
        item = await self._notifications.get()
        if item is None:
            # Keep the end marker for any later reader.
            self._notifications.put_nowait(None)
            raise TransportClosedError("app-server closed its output")
        return item

    async def serve(self) -> None:
        try:
            while line := await self._reader.readline():
                self._dispatch(line)
        finally:
            self._close()

    def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            log.warning("unparsable rpc line", extra={"line": line[:200]})
            return
        match message:
            case {"id": int() | str() as request_id, "method": str() as method}:
                task = asyncio.create_task(self._answer(request_id, method, _params(message)))
                self._answering.add(task)
                task.add_done_callback(self._answering.discard)
            case {"method": str() as method}:
                self._notifications.put_nowait(Notification(method, _params(message)))
            case {"id": int() | str() as request_id}:
                self._resolve(request_id, message)
            case _:
                log.warning("unexpected rpc message", extra={"line": line[:200]})

    def _resolve(self, request_id: RequestId, message: JsonObject) -> None:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        match message:
            case {"error": {"code": int() as code, "message": str() as text}}:
                future.set_exception(RpcError(code, text))
            case {"result": dict() as result}:
                future.set_result(result)
            case {"result": None}:
                future.set_result({})
            case _:
                future.set_exception(ProtocolError(f"malformed response: {message!r}"))

    async def _answer(self, request_id: RequestId, method: str, params: JsonObject) -> None:
        try:
            reply: JsonObject = {"id": request_id, "result": await self._handler(method, params)}
        except UnsupportedRequestError:
            reply = _error(request_id, METHOD_NOT_FOUND, f"unsupported: {method}")
        except MalformedRequestError as error:
            reply = _error(request_id, INVALID_PARAMS, str(error))
        except Exception:
            log.exception("server request failed", extra={"method": method})
            reply = _error(request_id, INTERNAL_ERROR, "agent-hub failed to handle the request")
        # The peer is gone; there is nobody left to answer.
        with contextlib.suppress(TransportClosedError):
            await self._send(reply)

    async def _send(self, message: JsonObject) -> None:
        if self._closed:
            raise TransportClosedError("app-server closed its output")
        self._writer.write(json.dumps(message).encode() + b"\n")
        await self._writer.drain()

    def _close(self) -> None:
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(TransportClosedError("app-server closed its output"))
        for task in self._answering:
            task.cancel()
        self._notifications.put_nowait(None)


def _params(message: JsonObject) -> JsonObject:
    params = message.get("params")
    return params if isinstance(params, dict) else {}


def _error(request_id: RequestId, code: int, message: str) -> JsonObject:
    return {"id": request_id, "error": {"code": code, "message": message}}
```

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/test_rpc.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: PASS. (If a `Future exception was never retrieved` log appears for the closed-transport test, it is harmless: the future is awaited by `call`.)

- [ ] **Step 5: Commit**

```bash
git add src/agent_hub/backends/rpc.py tests/test_rpc.py
git commit -m "JSON-RPC соединение поверх stdio дочернего процесса

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Codex protocol mapping `backends/codex_protocol.py`

**Files:**
- Create: `src/agent_hub/backends/codex_protocol.py`, `tests/test_codex_protocol.py`

**Interfaces:**
- Consumes: `common.HubTool`, descriptions/schemas, `TOOL_SUMMARY_LIMIT`; `rpc.JsonObject`, `rpc.Notification`, `rpc.ProtocolError`; `config.CodexSettings`.
- Produces: `TurnId = NewType("TurnId", str)`; `class Call(StrEnum)` (`INITIALIZE`, `INITIALIZED`, `ACCOUNT_READ`, `LOGIN`, `THREAD_START`, `THREAD_RESUME`, `TURN_START`, `TURN_STEER`); `class CodexTool(StrEnum)` (`SHELL = "shell"`, `PATCH = "patch"`, `WEB_SEARCH = "web_search"`); `class CodexAuth(StrEnum)` (`API_KEY`, `CHATGPT`, `OTHER`, `NOT_REQUIRED`, `MISSING`); `initialize_params() -> JsonObject`; `api_key_login_params(api_key: str) -> JsonObject`; `auth_state(result: JsonObject) -> CodexAuth`; `hub_tools() -> list[JsonObject]`; `open_thread(session: TopicSession, settings: CodexSettings) -> tuple[Call, JsonObject]`; `parse_thread_id(result) -> SessionId`; `turn_params(thread_id: SessionId, prompt: Prompt) -> JsonObject`; `steer_params(thread_id: SessionId, turn: TurnId, prompt: Prompt) -> JsonObject`; `parse_turn_id(result) -> TurnId`; `user_input(prompt) -> list[JsonObject]`; `Translation(events: tuple[AgentEvent, ...] = (), completed: TurnId | None = None)`; `TurnTracker(thread_id).translate(notification) -> Translation`.

- [ ] **Step 1: Write failing tests `tests/test_codex_protocol.py`**

```python
import base64
from pathlib import Path
from typing import Any

import pytest

from agent_hub.backends.codex_protocol import (
    Call,
    CodexAuth,
    Translation,
    TurnId,
    TurnTracker,
    auth_state,
    hub_tools,
    open_thread,
    parse_thread_id,
    parse_turn_id,
    steer_params,
    user_input,
)
from agent_hub.backends.rpc import Notification, ProtocolError
from agent_hub.config import CodexApproval, CodexSandbox, CodexSettings
from agent_hub.domain import (
    AssistantText,
    BackendKind,
    Failed,
    Finished,
    Image,
    ImageMediaType,
    Prompt,
    SessionId,
    ToolCall,
    TopicSession,
)

THREAD = SessionId("th-1")
SETTINGS = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, None)


def _session(cwd: Path, session_id: SessionId | None) -> TopicSession:
    # These functions ignore the backend kind; CODEX is added with the backend itself.
    return TopicSession(BackendKind.CLAUDE, cwd, session_id)


def _item(event: str, item: dict[str, Any]) -> Notification:
    return Notification(event, {"threadId": THREAD, "turnId": "turn-1", "item": item})


def _completed(turn: dict[str, Any]) -> Notification:
    return Notification("turn/completed", {"threadId": THREAD, "turn": turn})


def test_new_session_starts_a_thread_with_hub_tools(tmp_path: Path) -> None:
    method, params = open_thread(_session(tmp_path, None), SETTINGS)

    assert method is Call.THREAD_START
    assert params == {
        "cwd": str(tmp_path),
        "sandbox": "workspace-write",
        "approvalPolicy": "on-request",
        "approvalsReviewer": "user",
        "dynamicTools": hub_tools(),
    }
    assert [tool["name"] for tool in hub_tools()] == ["send_file", "ask_user"]


def test_saved_session_resumes_its_thread_with_the_model(tmp_path: Path) -> None:
    settings = CodexSettings("gpt-6.1-sol", CodexSandbox.READ_ONLY, CodexApproval.NEVER, None)

    method, params = open_thread(_session(tmp_path, THREAD), settings)

    assert method is Call.THREAD_RESUME
    assert params == {
        "cwd": str(tmp_path),
        "sandbox": "read-only",
        "approvalPolicy": "never",
        "approvalsReviewer": "user",
        "model": "gpt-6.1-sol",
        "threadId": "th-1",
    }


def test_ids_are_read_from_responses() -> None:
    assert parse_thread_id({"thread": {"id": "th-1", "preview": ""}}) == THREAD
    assert parse_turn_id({"turn": {"id": "turn-1", "items": []}}) == TurnId("turn-1")
    with pytest.raises(ProtocolError):
        parse_thread_id({"thread": {}})
    with pytest.raises(ProtocolError):
        parse_turn_id({})


def test_images_go_before_text_as_data_urls() -> None:
    prompt = Prompt("что на фото?", (Image(ImageMediaType.PNG, b"\x89P"),))

    assert user_input(prompt) == [
        {"type": "image", "url": f"data:image/png;base64,{base64.b64encode(b'\x89P').decode()}"},
        {"type": "text", "text": "что на фото?"},
    ]
    assert [item["type"] for item in user_input(Prompt("", prompt.images))] == ["image"]


def test_steer_names_the_expected_turn() -> None:
    assert steer_params(THREAD, TurnId("turn-1"), Prompt("ещё")) == {
        "threadId": "th-1",
        "expectedTurnId": "turn-1",
        "input": [{"type": "text", "text": "ещё"}],
    }


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {"type": "commandExecution", "id": "i", "command": "npm test", "cwd": "/w"},
            ToolCall("shell", "npm test"),
        ),
        (
            {
                "type": "fileChange",
                "id": "i",
                "status": "inProgress",
                "changes": [
                    {"path": "a.py", "kind": "add", "diff": ""},
                    {"path": "b.py", "kind": "update", "diff": ""},
                ],
            },
            ToolCall("patch", "a.py, b.py"),
        ),
        (
            {
                "type": "mcpToolCall",
                "id": "i",
                "server": "github",
                "tool": "get_issue",
                "arguments": {"n": 1},
                "status": "inProgress",
            },
            ToolCall("github/get_issue", '{"n": 1}'),
        ),
        ({"type": "webSearch", "id": "i", "query": "codex"}, ToolCall("web_search", "codex")),
    ],
)
def test_started_tool_items_become_tool_calls(item: dict[str, Any], expected: ToolCall) -> None:
    assert TurnTracker(THREAD).translate(_item("item/started", item)) == Translation((expected,))


@pytest.mark.parametrize(
    "item",
    [
        {"type": "dynamicToolCall", "id": "i", "tool": "send_file", "arguments": {}},
        {"type": "reasoning", "id": "i"},
        {"type": "agentMessage", "id": "i", "text": "ещё пишу"},
    ],
)
def test_other_started_items_are_silent(item: dict[str, Any]) -> None:
    assert TurnTracker(THREAD).translate(_item("item/started", item)) == Translation()


def test_completed_agent_message_becomes_text() -> None:
    tracker = TurnTracker(THREAD)
    message = {"type": "agentMessage", "id": "i", "text": "Готово"}

    assert tracker.translate(_item("item/completed", message)) == Translation(
        (AssistantText("Готово"),)
    )
    blank = {"type": "agentMessage", "id": "i", "text": "  "}
    assert tracker.translate(_item("item/completed", blank)) == Translation()


def test_completed_turn_finishes_with_thread_tokens() -> None:
    tracker = TurnTracker(THREAD)
    usage = {"total": {"totalTokens": 12345}, "last": {"totalTokens": 10}}
    tracker.translate(Notification("thread/tokenUsage/updated", {"tokenUsage": usage}))

    translation = tracker.translate(_completed({"id": "turn-1", "status": "completed"}))

    assert translation == Translation(
        (Finished(THREAD, None, None, tokens=12345),), TurnId("turn-1")
    )


@pytest.mark.parametrize(
    ("turn", "reason"),
    [
        ({"id": "turn-1", "status": "interrupted"}, "Ход Codex прерван"),
        ({"id": "turn-1", "status": "failed", "error": {"message": "401"}}, "401"),
        ({"id": "turn-1", "status": "failed"}, "Ход Codex завершился ошибкой"),
    ],
)
def test_unsuccessful_turn_fails(turn: dict[str, Any], reason: str) -> None:
    assert TurnTracker(THREAD).translate(_completed(turn)) == Translation(
        (Failed(reason),), TurnId("turn-1")
    )


def test_completion_without_a_turn_is_a_protocol_error() -> None:
    with pytest.raises(ProtocolError):
        TurnTracker(THREAD).translate(Notification("turn/completed", {}))


def test_unknown_notifications_are_ignored() -> None:
    notice = Notification("account/rateLimits/updated", {"x": 1})
    assert TurnTracker(THREAD).translate(notice) == Translation()


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"account": {"type": "apiKey"}, "requiresOpenaiAuth": True}, CodexAuth.API_KEY),
        (
            {"account": {"type": "chatgpt", "email": None, "planType": "plus"}},
            CodexAuth.CHATGPT,
        ),
        ({"account": {"type": "amazonBedrock"}, "requiresOpenaiAuth": False}, CodexAuth.OTHER),
        ({"account": None, "requiresOpenaiAuth": False}, CodexAuth.NOT_REQUIRED),
        ({"account": None, "requiresOpenaiAuth": True}, CodexAuth.MISSING),
    ],
)
def test_auth_state(result: dict[str, Any], expected: CodexAuth) -> None:
    assert auth_state(result) is expected
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_codex_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/agent_hub/backends/codex_protocol.py`**

```python
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
    TOKEN_USAGE = "thread/tokenUsage/updated"
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
    images: list[JsonObject] = [{"type": "image", "url": _data_url(image)} for image in prompt.images]
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
            return ToolCall(CodexTool.PATCH, truncate(", ".join(paths), TOOL_SUMMARY_LIMIT))
        case {"type": "mcpToolCall", "server": str() as server, "tool": str() as name}:
            arguments = json.dumps(item.get("arguments"), ensure_ascii=False)
            return ToolCall(f"{server}/{name}", truncate(arguments, TOOL_SUMMARY_LIMIT))
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
```

Note for the implementer: if mypy rejects `item.get(...)` after the mapping pattern on `object`, capture the value in the pattern instead: `case {"type": "mcpToolCall", "server": str() as server, "tool": str() as name, "arguments": arguments}` plus a second case without `"arguments"` using `{}`.

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/test_codex_protocol.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_hub/backends/codex_protocol.py tests/test_codex_protocol.py
git commit -m "Отображение протокола Codex app-server на события хаба

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: App-server requests `backends/codex_requests.py`

**Files:**
- Create: `src/agent_hub/backends/codex_requests.py`, `tests/test_codex_requests.py`

**Interfaces:**
- Consumes: `UserChannel`; `common.HubTool`, `ToolResult`/`ToolSuccess`/`ToolError`, `deliver_file`, `parse_questions`, `parse_option`, `TOOL_SUMMARY_LIMIT`; `rpc.JsonObject`, `MalformedRequestError`, `UnsupportedRequestError`; `codex_protocol.CodexTool`.
- Produces: `async def answer(channel: UserChannel, method: str, params: JsonObject) -> JsonObject` (raises `UnsupportedRequestError` / `MalformedRequestError`).

- [ ] **Step 1: Write failing tests `tests/test_codex_requests.py`**

```python
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


def _channel(decision: Allowed | Denied = Allowed()) -> FakeChannel:
    return FakeChannel(decision, Answered((("Какой формат?", "Кратко"),)))


@pytest.mark.parametrize(
    ("decision", "expected"), [(Allowed(), "accept"), (Denied("нет"), "decline")]
)
async def test_command_approval_follows_the_user(
    decision: Allowed | Denied, expected: str
) -> None:
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

    assert reply == {"success": False, "contentItems": [{"type": "inputText", "text": "вне директории"}]}


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
        (Question("Какой формат?", "Формат", (QuestionOption("Кратко", "Только суть"),), multi_select=False),)
    ]
    assert reply == {"answers": {"q1": {"answers": ["Кратко"]}}}


async def test_declined_native_questions_get_no_answers() -> None:
    channel = FakeChannel(Allowed(), Denied("не отвечу"))
    params = {**_BASE, "isBlocking": True, "questions": [{"id": "q1", "header": "", "question": "Да?"}]}

    assert await answer(channel, USER_INPUT, params) == {"answers": {}}


@pytest.mark.parametrize(
    "questions", [[], [{"id": "q1", "header": "h"}], [{"id": "q1", "header": "h", "question": "?", "options": [{}]}]]
)
async def test_malformed_native_questions(questions: list[Any]) -> None:
    with pytest.raises(MalformedRequestError):
        await answer(_channel(), USER_INPUT, {**_BASE, "isBlocking": True, "questions": questions})


async def test_unknown_request_is_unsupported() -> None:
    with pytest.raises(UnsupportedRequestError):
        await answer(_channel(), "mcpServer/elicitation/request", {})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_codex_requests.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/agent_hub/backends/codex_requests.py`**

```python
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
```

Note: a question with `"options": null` matches the second case (no list), so it is asked without buttons.

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/test_codex_requests.py -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent_hub/backends/codex_requests.py tests/test_codex_requests.py
git commit -m "Ответы на запросы Codex: одобрения, инструменты хаба, вопросы

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `CodexBackend`, process launcher and wiring

**Files:**
- Create: `src/agent_hub/backends/codex.py`, `tests/test_codex_backend.py`
- Modify: `pyproject.toml`, `uv.lock`, `src/agent_hub/domain.py`, `src/agent_hub/__main__.py`, `tests/test_codex_protocol.py`

**Interfaces:**
- Consumes: everything from Tasks 3–5.
- Produces: `BackendKind.CODEX = "codex"`; `Launcher = Callable[[RequestHandler], AbstractAsyncContextManager[Connection]]`; `app_server(handler) -> AsyncIterator[Connection]` (async context manager); `class CodexThread(Protocol)` (`start_turn(prompt) -> TurnId`, `steer(turn, prompt) -> None`, `notification() -> Notification`); `async def converse(thread, tracker, prompt, inbox) -> AsyncGenerator[AgentEvent, None]`; `CodexBackend(settings: CodexSettings, launch: Launcher = app_server)`; `async def probe_auth(settings: CodexSettings, launch: Launcher = app_server) -> CodexAuth`; constants `NOT_LOGGED_IN`, `APP_SERVER_FAILED`; `__main__.check_codex(settings: CodexSettings) -> None`.

- [ ] **Step 1: Add the binary dependency**

Run: `uv add "openai-codex-cli-bin>=0.160.0"`
Then add to `pyproject.toml` after `[tool.mypy]`:

```toml
[[tool.mypy.overrides]]
# Annotated, but ships without a py.typed marker.
module = ["codex_cli_bin"]
follow_untyped_imports = true
```

Verify: `uv run python -c "import codex_cli_bin; print(codex_cli_bin.bundled_codex_path())"` prints a path to `codex`/`codex.exe`.

- [ ] **Step 2: Write failing tests `tests/test_codex_backend.py`**

```python
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from agent_hub.backends.codex import (
    APP_SERVER_FAILED,
    NOT_LOGGED_IN,
    CodexBackend,
    Launcher,
    converse,
    probe_auth,
)
from agent_hub.backends.codex_protocol import CodexAuth, TurnId, TurnTracker
from agent_hub.backends.rpc import (
    Connection,
    JsonObject,
    Notification,
    RequestHandler,
    RpcError,
    TransportClosedError,
)
from agent_hub.config import CodexApproval, CodexSandbox, CodexSettings
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    Answered,
    AssistantText,
    BackendKind,
    Failed,
    Finished,
    Prompt,
    SessionId,
    SessionStarted,
    TopicSession,
)
from tests.fakes import FakeChannel

THREAD = SessionId("th-1")
SETTINGS = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, None)
CHATGPT = {"account": {"type": "chatgpt", "email": None, "planType": "plus"}}


def _completed(turn_id: str, status: str = "completed") -> Notification:
    return Notification("turn/completed", {"threadId": THREAD, "turn": {"id": turn_id, "status": status}})


def _message(text: str) -> Notification:
    item = {"type": "agentMessage", "id": "m", "text": text}
    return Notification("item/completed", {"threadId": THREAD, "turnId": "turn-1", "item": item})


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class FakeThread:
    def __init__(self, steer_error: RpcError | None = None) -> None:
        self.steer_error = steer_error
        self.incoming: asyncio.Queue[Notification | None] = asyncio.Queue()
        self.turns: list[str] = []
        self.steered: list[tuple[TurnId, str]] = []

    async def start_turn(self, prompt: Prompt) -> TurnId:
        self.turns.append(prompt.text)
        return TurnId(f"turn-{len(self.turns)}")

    async def steer(self, turn: TurnId, prompt: Prompt) -> None:
        if self.steer_error is not None:
            raise self.steer_error
        self.steered.append((turn, prompt.text))

    async def notification(self) -> Notification:
        notification = await self.incoming.get()
        if notification is None:
            raise TransportClosedError("closed")
        return notification

    def feed(self, *notifications: Notification | None) -> None:
        for notification in notifications:
            self.incoming.put_nowait(notification)


async def _collect(thread: FakeThread, inbox: asyncio.Queue[Prompt]) -> list[AgentEvent]:
    return [
        event
        async for event in converse(thread, TurnTracker(THREAD), Prompt("задача"), inbox)
    ]


async def test_turn_relays_text_and_finishes() -> None:
    thread = FakeThread()
    thread.feed(_message("Готово"), _completed("turn-1"))

    events = await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)

    assert events == [AssistantText("Готово"), Finished(THREAD, None, None)]
    assert thread.turns == ["задача"]


async def test_prompt_during_a_turn_steers_it() -> None:
    thread = FakeThread()
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("ещё"))
    await _settle()
    thread.feed(_completed("turn-1"))

    events = await asyncio.wait_for(run, 1)
    assert thread.steered == [(TurnId("turn-1"), "ещё")]
    assert events == [Finished(THREAD, None, None)]


async def test_rejected_steer_starts_a_new_turn() -> None:
    thread = FakeThread(steer_error=RpcError(-32600, "no active turn"))
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("ещё"))
    await _settle()
    assert thread.turns == ["задача", "ещё"]

    thread.feed(_completed("turn-1"))
    await _settle()
    assert not run.done()

    thread.feed(_completed("turn-2"))
    events = await asyncio.wait_for(run, 1)
    assert events == [Finished(THREAD, None, None), Finished(THREAD, None, None)]


async def test_failed_turn_ends_the_session() -> None:
    thread = FakeThread()
    thread.feed(_completed("turn-1", "interrupted"))

    events = await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)

    assert events == [Failed("Ход Codex прерван")]


async def test_closed_transport_propagates() -> None:
    thread = FakeThread()
    thread.feed(None)

    with pytest.raises(TransportClosedError):
        await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)


class FakeConnection:
    def __init__(self, responses: dict[str, JsonObject | RpcError]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, JsonObject]] = []
        self.notified: list[str] = []
        self.incoming: asyncio.Queue[Notification] = asyncio.Queue()

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        self.calls.append((method, params))
        response = self.responses[method]
        if isinstance(response, RpcError):
            raise response
        return response

    async def notify(self, method: str) -> None:
        self.notified.append(method)

    async def notification(self) -> Notification:
        return await self.incoming.get()


def _launcher(connection: FakeConnection) -> Launcher:
    @asynccontextmanager
    async def launch(_handler: RequestHandler) -> AsyncIterator[Connection]:
        yield connection

    return launch


def _responses(**overrides: JsonObject | RpcError) -> dict[str, JsonObject | RpcError]:
    return {
        "initialize": {},
        "account/read": CHATGPT,
        "thread/start": {"thread": {"id": "th-1"}},
        "thread/resume": {"thread": {"id": "th-1"}},
        "turn/start": {"turn": {"id": "turn-1"}},
        **{key.replace("_", "/"): value for key, value in overrides.items()},
    }


async def _run(backend: CodexBackend, session: TopicSession) -> list[AgentEvent]:
    channel = FakeChannel(Allowed(), Answered(()))
    events = backend.run(session, Prompt("задача"), channel, asyncio.Queue())
    return [event async for event in events]


async def test_new_session_starts_a_thread_and_runs_a_turn(tmp_path: Path) -> None:
    connection = FakeConnection(_responses())
    connection.incoming.put_nowait(_completed("turn-1"))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await asyncio.wait_for(_run(backend, TopicSession(BackendKind.CODEX, tmp_path, None)), 1)

    assert events == [SessionStarted(THREAD), Finished(THREAD, None, None)]
    assert [method for method, _ in connection.calls] == [
        "initialize",
        "account/read",
        "thread/start",
        "turn/start",
    ]
    assert connection.notified == ["initialized"]


async def test_missing_login_fails_before_opening_a_thread(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(account_read={"account": None, "requiresOpenaiAuth": True}))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [Failed(NOT_LOGGED_IN)]


async def test_resume_failure_suggests_reset(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(thread_resume=RpcError(-32600, "no rollout found")))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, THREAD))

    assert len(events) == 1
    assert isinstance(events[0], Failed)
    assert "no rollout found" in events[0].reason
    assert "/reset" in events[0].reason


async def test_spawn_failure_fails_the_turn(tmp_path: Path) -> None:
    @asynccontextmanager
    async def broken(_handler: RequestHandler) -> AsyncIterator[Connection]:
        raise FileNotFoundError("codex")
        yield  # pragma: no cover

    events = await _run(CodexBackend(SETTINGS, broken), TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [Failed(APP_SERVER_FAILED)]


async def test_probe_logs_in_with_the_configured_key() -> None:
    connection = FakeConnection(
        _responses(account_read={"account": {"type": "apiKey"}}, account_login_start={"type": "apiKey"})
    )
    settings = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, "sk-test")

    assert await probe_auth(settings, _launcher(connection)) is CodexAuth.API_KEY
    assert ("account/login/start", {"type": "apiKey", "apiKey": "sk-test"}) in connection.calls


async def test_real_app_server_reports_missing_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    assert await asyncio.wait_for(probe_auth(SETTINGS), 60) is CodexAuth.MISSING


async def test_missing_login_fails_with_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    session = TopicSession(BackendKind.CODEX, tmp_path, None)

    events = await asyncio.wait_for(_run(CodexBackend(SETTINGS), session), 60)

    assert events == [Failed(NOT_LOGGED_IN)]


async def test_real_app_server_accepts_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    settings = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, "sk-test")

    assert await asyncio.wait_for(probe_auth(settings), 60) is CodexAuth.API_KEY
```

Notes: `_responses(account_read=...)` maps `account_read` → `account/read`, `account_login_start` → `account/login/start`, `thread_resume` → `thread/resume`. The three `real_app_server` tests spawn the bundled binary (offline, ~1 s each; verified on 0.160.0). In `test_spawn_failure_fails_the_turn` the unreachable `yield` keeps the function an async generator; if ruff flags it, replace the body with a class-based context manager whose `__aenter__` raises.

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_codex_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_hub.backends.codex'`.

- [ ] **Step 4: Implement `src/agent_hub/backends/codex.py`**

```python
"""OpenAI Codex backend via `codex app-server`, JSON-RPC over the child's stdio."""

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol, final

import codex_cli_bin

from agent_hub.backends import Inbox, UserChannel
from agent_hub.backends.codex_protocol import (
    Call,
    CodexAuth,
    TurnId,
    TurnTracker,
    api_key_login_params,
    auth_state,
    initialize_params,
    open_thread,
    parse_thread_id,
    parse_turn_id,
    steer_params,
    turn_params,
)
from agent_hub.backends.codex_requests import answer
from agent_hub.backends.rpc import (
    Connection,
    JsonObject,
    Notification,
    ProtocolError,
    RequestHandler,
    RpcConnection,
    RpcError,
    TransportClosedError,
    UnsupportedRequestError,
)
from agent_hub.config import CodexSettings
from agent_hub.domain import AgentEvent, Failed, Prompt, SessionId, SessionStarted, TopicSession

log = logging.getLogger(__name__)

# A whole turn item (a diff, a command's output) arrives as one stdout line.
LINE_LIMIT = 64 * 2**20
# Handshake and thread/turn control only; a turn itself runs until it ends or /stop.
REQUEST_TIMEOUT_SECONDS = 60
NOT_LOGGED_IN = (
    "Codex не авторизован: выполните `codex login --device-auth` или задайте OPENAI_API_KEY "
    "и перезапустите бота"
)
APP_SERVER_FAILED = "Codex app-server завершился, подробности в логе сервиса"

Launcher = Callable[[RequestHandler], AbstractAsyncContextManager[Connection]]


@asynccontextmanager
async def app_server(handler: RequestHandler) -> AsyncIterator[Connection]:
    """A `codex app-server` child answering server requests with `handler`; killed on exit."""
    process = await asyncio.create_subprocess_exec(
        codex_cli_bin.bundled_codex_path(),
        "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_child_env(),
        limit=LINE_LIMIT,
    )
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr
    if stdin is None or stdout is None or stderr is None:
        process.kill()
        await process.wait()
        raise TransportClosedError("app-server started without pipes")
    connection = RpcConnection(stdout, stdin, handler)
    tasks = [asyncio.create_task(connection.serve()), asyncio.create_task(_log_stderr(stderr))]
    try:
        yield connection
    finally:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                log.warning("codex transport failed", exc_info=result)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    tools = codex_cli_bin.bundled_path_dir()
    if tools is not None:
        # Bundled helpers such as ripgrep, put on PATH the way the official SDK does.
        env["PATH"] = os.pathsep.join([str(tools), env.get("PATH", "")])
    return env


async def _log_stderr(stream: asyncio.StreamReader) -> None:
    while line := await stream.readline():
        log.info("codex stderr", extra={"line": line.decode(errors="replace").rstrip()})


async def _call(connection: Connection, method: Call, params: JsonObject) -> JsonObject:
    async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
        return await connection.request(method, params)


async def handshake(connection: Connection) -> None:
    await _call(connection, Call.INITIALIZE, initialize_params())
    await connection.notify(Call.INITIALIZED)


class CodexThread(Protocol):
    """The part of an app-server thread a session needs."""

    async def start_turn(self, prompt: Prompt) -> TurnId: ...

    async def steer(self, turn: TurnId, prompt: Prompt) -> None: ...

    async def notification(self) -> Notification: ...


@final
class RpcThread:
    def __init__(self, connection: Connection, thread_id: SessionId) -> None:
        self._connection = connection
        self._thread_id = thread_id

    async def start_turn(self, prompt: Prompt) -> TurnId:
        params = turn_params(self._thread_id, prompt)
        return parse_turn_id(await _call(self._connection, Call.TURN_START, params))

    async def steer(self, turn: TurnId, prompt: Prompt) -> None:
        params = steer_params(self._thread_id, turn, prompt)
        await _call(self._connection, Call.TURN_STEER, params)

    async def notification(self) -> Notification:
        return await self._connection.notification()


class CodexBackend:
    def __init__(self, settings: CodexSettings, launch: Launcher = app_server) -> None:
        self._settings = settings
        self._launch = launch

    async def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel, inbox: Inbox
    ) -> AsyncGenerator[AgentEvent, None]:
        async def handle(method: str, params: JsonObject) -> JsonObject:
            return await answer(channel, method, params)

        try:
            async with self._launch(handle) as connection:
                await handshake(connection)
                account = await _call(connection, Call.ACCOUNT_READ, {})
                if auth_state(account) is CodexAuth.MISSING:
                    yield Failed(NOT_LOGGED_IN)
                    return
                method, params = open_thread(session, self._settings)
                try:
                    thread_id = parse_thread_id(await _call(connection, method, params))
                except RpcError as error:
                    yield Failed(_open_failure(session, error))
                    return
                yield SessionStarted(thread_id)
                thread = RpcThread(connection, thread_id)
                async for event in converse(thread, TurnTracker(thread_id), prompt, inbox):
                    yield event
        except RpcError as error:
            yield Failed(f"Codex: {error.message}")
        except (TransportClosedError, ProtocolError, TimeoutError, OSError):
            log.warning("codex session broke", exc_info=True)
            yield Failed(APP_SERVER_FAILED)


def _open_failure(session: TopicSession, error: RpcError) -> str:
    if session.session_id is None:
        return f"Codex не начал сессию: {error.message}"
    return (
        f"Codex не смог продолжить сессию {session.session_id}: {error.message}. "
        "/reset — начать заново"
    )


async def converse(
    thread: CodexThread, tracker: TurnTracker, prompt: Prompt, inbox: Inbox
) -> AsyncGenerator[AgentEvent, None]:
    """Relay a thread until no turn is running and nothing is queued.

    Prompts sent meanwhile join the running turn; one that arrives as the turn ends starts
    the next turn instead.
    """
    active: TurnId | None = await thread.start_turn(prompt)
    next_notification = asyncio.ensure_future(thread.notification())
    next_prompt = asyncio.ensure_future(inbox.get())
    try:
        # A prompt already taken from the inbox must be delivered even if no turn runs.
        while active is not None or next_prompt.done():
            done, _ = await asyncio.wait(
                {next_notification, next_prompt}, return_when=asyncio.FIRST_COMPLETED
            )
            if next_prompt in done:
                active = await _deliver(thread, active, next_prompt.result())
                next_prompt = asyncio.ensure_future(inbox.get())
            if next_notification in done:
                translation = tracker.translate(next_notification.result())
                for event in translation.events:
                    yield event
                if translation.completed is not None and translation.completed == active:
                    active = None
                next_notification = asyncio.ensure_future(thread.notification())
    finally:
        next_notification.cancel()
        next_prompt.cancel()


async def _deliver(thread: CodexThread, active: TurnId | None, prompt: Prompt) -> TurnId:
    if active is None:
        return await thread.start_turn(prompt)
    try:
        await thread.steer(active, prompt)
    except RpcError:
        # The turn ended before the steer reached it; its completion is still on the way.
        return await thread.start_turn(prompt)
    return active


async def probe_auth(settings: CodexSettings, launch: Launcher = app_server) -> CodexAuth:
    """Log in with the configured API key, if any, and report how Codex is authenticated."""
    async with launch(_refuse) as connection:
        await handshake(connection)
        if settings.api_key is not None:
            await _call(connection, Call.LOGIN, api_key_login_params(settings.api_key))
        return auth_state(await _call(connection, Call.ACCOUNT_READ, {}))


async def _refuse(method: str, _params: JsonObject) -> JsonObject:
    raise UnsupportedRequestError(method)
```

- [ ] **Step 5: Register the backend**

`domain.py`:
```python
class BackendKind(StrEnum):
    CLAUDE = "claude"
    CODEX = "codex"
```

`__main__.py` — imports `asyncio`, `from agent_hub.backends.codex import CodexBackend, probe_auth`, `from agent_hub.backends.codex_protocol import CodexAuth`, `from agent_hub.backends.rpc import ProtocolError, RpcError, TransportClosedError`, `from agent_hub.config import CodexSettings`; then:

```python
def build_backend(kind: BackendKind, settings: Settings) -> AgentBackend:
    match kind:
        case BackendKind.CLAUDE:
            return ClaudeBackend(settings.claude, settings.background_timeout_seconds)
        case BackendKind.CODEX:
            return CodexBackend(settings.codex)
        case _:
            assert_never(kind)


def check_codex(settings: CodexSettings) -> None:
    """Report at startup whether Codex topics can work; never stops the hub."""
    # A private loop: asyncio.run would leave no current loop for run_polling.
    loop = asyncio.new_event_loop()
    try:
        auth = loop.run_until_complete(probe_auth(settings))
    except (OSError, RpcError, TransportClosedError, ProtocolError, TimeoutError):
        log.warning("codex unavailable", exc_info=True)
        return
    finally:
        loop.close()
    match auth:
        case CodexAuth.MISSING:
            log.warning("codex not logged in")
        case CodexAuth.API_KEY | CodexAuth.CHATGPT | CodexAuth.OTHER | CodexAuth.NOT_REQUIRED:
            log.info("codex ready", extra={"auth": auth.value})
        case _:
            assert_never(auth)
```

In `main()` after `backends = ...`: `check_codex(settings.codex)`; add `"default_backend": settings.default_backend.value` to the `starting` log extra.

- [ ] **Step 6: Run tests and checks**

Run: `uv run pytest -q && uv run ruff format . && uv run ruff check . && uv run mypy`
Expected: PASS, including the three tests that start the real binary.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src tests
git commit -m "Бэкенд Codex через codex app-server

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `/backend` command

**Files:**
- Modify: `src/agent_hub/commands.py`, `src/agent_hub/bot.py`
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: `BackendKind` with `CODEX`; `Settings.default_backend`.
- Produces: `ShowBackend()`, `UnknownBackend(name: str)`, `BackendArgs = BackendKind | ShowBackend | UnknownBackend`, `parse_backend_args(args: Sequence[str]) -> BackendArgs`; bot command `/backend`.

- [ ] **Step 1: Failing tests** — append to `tests/test_commands.py` (import `ShowBackend, UnknownBackend, parse_backend_args`):

```python
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ShowBackend()),
        (["codex"], BackendKind.CODEX),
        (["Claude"], BackendKind.CLAUDE),
        (["gemini"], UnknownBackend("gemini")),
        (["codex", "extra"], UnknownBackend("codex extra")),
    ],
)
def test_parse_backend_args(args: list[str], expected: object) -> None:
    assert parse_backend_args(args) == expected


def test_new_with_codex_and_a_codex_default() -> None:
    assert parse_new_args(["codex", "shop"], BackendKind.CLAUDE) == NewSessionArgs(
        BackendKind.CODEX, "shop"
    )
    assert parse_new_args(["shop"], BackendKind.CODEX) == NewSessionArgs(BackendKind.CODEX, "shop")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_commands.py -q`
Expected: FAIL — `ImportError: cannot import name 'ShowBackend'`.

- [ ] **Step 3: Implement in `commands.py`**

```python
@final
@dataclass(frozen=True, slots=True)
class ShowBackend:
    pass


@final
@dataclass(frozen=True, slots=True)
class UnknownBackend:
    name: str


BackendArgs = BackendKind | ShowBackend | UnknownBackend


def parse_backend_args(args: Sequence[str]) -> BackendArgs:
    """`/backend [name]`: no name shows the current backend."""
    match args:
        case []:
            return ShowBackend()
        case [name]:
            try:
                return BackendKind(name.lower())
            except ValueError:
                return UnknownBackend(name)
        case _:
            return UnknownBackend(" ".join(args))
```

- [ ] **Step 4: Implement in `bot.py`**

Import `ShowBackend, UnknownBackend, parse_backend_args`. In `HELP` add after the `/new` line:

```
/backend [claude|codex] — сменить агента в этой теме (сброс контекста)
```

Register after the `cwd` handler: `app.add_handler(CommandHandler("backend", self._backend, filters=fresh))`. Add after `_cwd`:

```python
    async def _backend(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        key = await self._topic_or_hint(update)
        if key is None:
            return
        sender = TelegramSender(context.bot)
        current = self._session(key)
        args = parse_backend_args(context.args or [])
        match args:
            case ShowBackend():
                await sender.text(key, _describe("ℹ️ Текущая сессия", current))
            case UnknownBackend(name):
                known = ", ".join(kind.value for kind in BackendKind)
                await sender.text(key, f"⚠️ Неизвестный бэкенд {name}. Доступны: {known}")
            case BackendKind():
                if await self._refuse_if_running(key, context.bot):
                    return
                # One agent cannot continue another's session.
                session = TopicSession(args, current.cwd, None)
                self._store.put(key, session)
                await sender.text(key, _describe("🔀 Бэкенд изменён", session))
            case _:
                assert_never(args)
```

- [ ] **Step 5: Run checks**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "Команда /backend: смена агента в теме

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docker, CI and documentation

**Files:**
- Modify: `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml`, `.env.example`, `README.md`

- [ ] **Step 1: Dockerfile** — in the `base` stage:
  - `mkdir -p` and `chown` lists gain `/home/app/.codex`;
  - after the `claude` symlink:
```dockerfile
# The codex-cli-bin package ships the Codex binary; expose it for `codex login` inside the container.
RUN ln -s /app/.venv/lib/python3.12/site-packages/codex_cli_bin/bin/codex /usr/local/bin/codex
```
  - `ENV` gains `CODEX_HOME=/home/app/.codex \`.
  - Header comment of `base`: `# Bot + Claude Code + Codex + git: enough for Python and plain-text projects.`

- [ ] **Step 2: compose.yaml** — under `environment`:
```yaml
      # Codex's own sandbox cannot start inside a container; the container is the sandbox.
      AGENT_HUB_CODEX_SANDBOX: danger-full-access
```
under `volumes` of the service: `- codex:/home/app/.codex`; top-level `volumes:` gains `codex:`.

- [ ] **Step 3: CI** — after the `claude --version` step in `.github/workflows/ci.yml`:
```yaml
      - run: docker run --rm --entrypoint codex agent-hub:ci --version
```

- [ ] **Step 4: `.env.example`** — replace the Claude auth block with:
```
# Авторизация агентов в контейнере, если не входить внутри него (см. README).
# ANTHROPIC_API_KEY=sk-ant-...
# CLAUDE_CODE_OAUTH_TOKEN=...
# Ключ OpenAI для Codex; при старте бот сохраняет его в CODEX_HOME, он важнее входа по подписке.
# OPENAI_API_KEY=sk-...
```
and append to the optional section:
```
# Агент новых тем: claude | codex
# AGENT_HUB_DEFAULT_BACKEND=claude

# Модель Codex; по умолчанию — модель из настроек Codex.
# AGENT_HUB_CODEX_MODEL=gpt-6.1-sol

# read-only | workspace-write | danger-full-access (в Docker задано в compose.yaml)
# AGENT_HUB_CODEX_SANDBOX=workspace-write

# Когда Codex спрашивает разрешение: untrusted | on-request | never
# AGENT_HUB_CODEX_APPROVAL=on-request
```

- [ ] **Step 5: README.md**
  - Intro: replace "Сейчас поддерживается один бэкенд — **Claude** …" paragraph with: "Поддерживаются два агента — **Claude** (через [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)) и **Codex** (через `codex app-server`). Агент выбирается для каждой темы: `/new codex <путь>` или `/backend codex`." Replace "получили новую сессию Claude Code" with "получили новую сессию агента"; diagram lines may show one `сессия Codex`.
  - Requirements table: add row `| [Codex](https://developers.openai.com/codex), вход (`codex login`) или `OPENAI_API_KEY` | Только для тем с Codex; бинарник ставится вместе с проектом |`.
  - Commands table: add `| `/backend [claude\|codex]` | Показать или сменить агента темы (контекст сбрасывается) |`; example block gains `/backend codex`.
  - Configuration table: add rows `AGENT_HUB_DEFAULT_BACKEND` (нет, `claude`, агент новых тем), `AGENT_HUB_CODEX_MODEL` (нет, из настроек Codex), `AGENT_HUB_CODEX_SANDBOX` (нет, `workspace-write`), `AGENT_HUB_CODEX_APPROVAL` (нет, `on-request`), `OPENAI_API_KEY` (нет, —, ключ для Codex; при старте сохраняется в `CODEX_HOME` и важнее входа по подписке). After the Claude modes table add:

```markdown
Режимы Codex:

| `AGENT_HUB_CODEX_SANDBOX` | Что разрешено командам |
|---|---|
| `read-only` | Только чтение |
| `workspace-write` | Запись в рабочей директории. **Рекомендуется** на хосте |
| `danger-full-access` | Без ограничений. Только в изолированном окружении (по умолчанию в Docker) |

| `AGENT_HUB_CODEX_APPROVAL` | Когда спрашивает |
|---|---|
| `untrusted` | Почти перед каждой командой |
| `on-request` | Когда нужно выйти за песочницу или модель сочтёт нужным. **Рекомендуется** |
| `never` | Никогда |
```

  - Docker section: rename "### 3. Авторизуйте Claude" → "### 3. Авторизуйте агентов", keep the Claude list under "**Claude** — один из вариантов:" and add:

```markdown
**Codex** — один из вариантов:

- **Вход по подписке ChatGPT** — один раз, данные сохраняются в томе `codex`:

  ```bash
  docker compose run --rm -it --entrypoint codex agent-hub login --device-auth
  ```

- **API-ключ:** `OPENAI_API_KEY=sk-...` в `.env`; при старте бот передаёт его Codex.

Без входа темы с Codex отвечают ошибкой, остальное работает; в логе при старте —
`codex not logged in`.
```

  - "Что где хранится": row `| /home/app/.codex | том codex | Логин, настройки и треды Codex |`.
  - "Как это устроено" tree: under `backends/` add `common.py # общие инструменты хаба (send_file, ask_user)`, `rpc.py # JSON-RPC поверх stdio`, `codex.py # бэкенд Codex: процесс app-server и сессия`, `codex_protocol.py # сообщения app-server ↔ события хаба`, `codex_requests.py # одобрения, инструменты и вопросы Codex`. In step 2 of the flow mention that Codex has no background tasks and a message sent during a turn joins it (`turn/steer`).

- [ ] **Step 6: Verify the image**

Run: `docker build --target base -t agent-hub:ci . && docker run --rm --entrypoint codex agent-hub:ci --version && docker run --rm --entrypoint claude agent-hub:ci --version`
Expected: `codex-cli 0.160.0` and the Claude version.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile compose.yaml .github/workflows/ci.yml .env.example README.md
git commit -m "Codex в Docker-образе, CI и документации

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Manual end-to-end check in Telegram

Requires a real Codex login (ChatGPT subscription or `OPENAI_API_KEY`). No code changes; record outcomes in the PR description.

- [ ] Start the bot (`uv run agent-hub` or `docker compose up -d`); the log shows `codex ready auth=chatgpt|apiKey`.
- [ ] `/new codex <проект>` → «🆕 Новая сессия», `backend: codex`.
- [ ] Task with a command (e.g. «запусти тесты») → `🔧 shell` line, approval buttons; «Запретить» → Codex continues without running it; «Разрешить» → command runs.
- [ ] Task that edits a file → `🔧 patch` line with paths; approval if the sandbox requires it.
- [ ] «пришли README файлом» → document arrives in the topic.
- [ ] Ask Codex to clarify via `ask_user` → question buttons; answer → Codex uses it.
- [ ] Send a photo with a question → the answer refers to the picture.
- [ ] While a turn runs, send another message → it is taken into the same turn (one `✅ Готово`).
- [ ] Final line shows `токенов в сессии: …`; after bot restart the next message continues the same thread (`/status` shows the same session id).
- [ ] `/backend claude` while a task runs → refused; after it ends → switched, session reset; `/backend` alone shows the current session.
- [ ] Remove the Codex login, restart → log `codex not logged in`; message in a Codex topic → the `NOT_LOGGED_IN` text; Claude topics still work.
