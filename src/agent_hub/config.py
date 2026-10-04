"""Settings parsed once from environment variables."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import final

from agent_hub.domain import BackendKind

PREFIX = "AGENT_HUB_"
# Codex's own variable name, so an existing key works without renaming.
OPENAI_API_KEY = "OPENAI_API_KEY"
DEFAULT_STATE_FILE = Path("~/.local/state/agent-hub/topics.json")
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 600
DEFAULT_BACKGROUND_TIMEOUT_SECONDS = 1800


class ConfigError(Exception):
    pass


class PermissionMode(StrEnum):
    """Claude Code permission modes exposed to the operator."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"


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


def load_settings(env: Mapping[str, str]) -> Settings:
    workspace_root = Path(_required(env, "WORKSPACE_ROOT")).expanduser().resolve()
    if not workspace_root.is_dir():
        raise ConfigError(f"{PREFIX}WORKSPACE_ROOT is not a directory: {workspace_root}")
    return Settings(
        telegram_token=_required(env, "TELEGRAM_TOKEN"),
        chat_id=_parse_int(_required(env, "CHAT_ID"), "CHAT_ID"),
        allowed_user_ids=_parse_user_ids(_required(env, "ALLOWED_USER_IDS")),
        workspace_root=workspace_root,
        state_file=Path(_optional(env, "STATE_FILE") or DEFAULT_STATE_FILE).expanduser(),
        approval_timeout_seconds=_parse_positive_int(
            _optional(env, "APPROVAL_TIMEOUT_SECONDS"),
            "APPROVAL_TIMEOUT_SECONDS",
            DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        ),
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
    )


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(PREFIX + name, "").strip()
    return value or None


def _required(env: Mapping[str, str], name: str) -> str:
    value = _optional(env, name)
    if value is None:
        raise ConfigError(f"{PREFIX}{name} is required")
    return value


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigError(f"{PREFIX}{name} must be an integer, got {raw!r}") from error


def _parse_positive_int(raw: str | None, name: str, default: int) -> int:
    if raw is None:
        return default
    value = _parse_int(raw, name)
    if value <= 0:
        raise ConfigError(f"{PREFIX}{name} must be positive, got {value}")
    return value


def _parse_user_ids(raw: str) -> frozenset[int]:
    ids = frozenset(
        _parse_int(part.strip(), "ALLOWED_USER_IDS") for part in raw.split(",") if part.strip()
    )
    if not ids:
        raise ConfigError(f"{PREFIX}ALLOWED_USER_IDS must list at least one user id")
    return ids


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


def _parse_budget(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ConfigError(f"{PREFIX}CLAUDE_MAX_BUDGET_USD must be a number, got {raw!r}") from error
    if not value.is_finite() or value <= 0:
        raise ConfigError(f"{PREFIX}CLAUDE_MAX_BUDGET_USD must be positive, got {raw!r}")
    return value
