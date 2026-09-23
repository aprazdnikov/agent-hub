"""Settings parsed once from environment variables."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import final

PREFIX = "AGENT_HUB_"
DEFAULT_STATE_FILE = Path("~/.local/state/agent-hub/topics.json")
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 600


class ConfigError(Exception):
    pass


class PermissionMode(StrEnum):
    """Claude Code permission modes exposed to the operator."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"


@final
@dataclass(frozen=True, slots=True)
class ClaudeSettings:
    permission_mode: PermissionMode
    model: str | None
    max_budget_usd: Decimal | None


@final
@dataclass(frozen=True, slots=True)
class Settings:
    telegram_token: str
    chat_id: int
    allowed_user_ids: frozenset[int]
    workspace_root: Path
    state_file: Path
    approval_timeout_seconds: int
    claude: ClaudeSettings


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
        claude=ClaudeSettings(
            permission_mode=_parse_permission_mode(_optional(env, "CLAUDE_PERMISSION_MODE")),
            model=_optional(env, "CLAUDE_MODEL"),
            max_budget_usd=_parse_budget(_optional(env, "CLAUDE_MAX_BUDGET_USD")),
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


def _parse_permission_mode(raw: str | None) -> PermissionMode:
    if raw is None:
        return PermissionMode.DEFAULT
    try:
        return PermissionMode(raw)
    except ValueError as error:
        allowed = ", ".join(mode.value for mode in PermissionMode)
        raise ConfigError(
            f"{PREFIX}CLAUDE_PERMISSION_MODE must be one of: {allowed}; got {raw!r}"
        ) from error


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
