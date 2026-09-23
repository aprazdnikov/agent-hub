from decimal import Decimal
from pathlib import Path

import pytest

from agent_hub.config import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ConfigError,
    PermissionMode,
    load_settings,
)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {
        "AGENT_HUB_TELEGRAM_TOKEN": "123:abc",
        "AGENT_HUB_CHAT_ID": "-1001234567890",
        "AGENT_HUB_ALLOWED_USER_IDS": "111, 222",
        "AGENT_HUB_WORKSPACE_ROOT": str(tmp_path),
    }


def test_minimal_env_uses_defaults(env: dict[str, str], tmp_path: Path) -> None:
    settings = load_settings(env)

    assert settings.chat_id == -1001234567890
    assert settings.allowed_user_ids == frozenset({111, 222})
    assert settings.workspace_root == tmp_path.resolve()
    assert settings.approval_timeout_seconds == DEFAULT_APPROVAL_TIMEOUT_SECONDS
    assert settings.claude.permission_mode is PermissionMode.DEFAULT
    assert settings.claude.model is None
    assert settings.claude.max_budget_usd is None


def test_optional_values_are_parsed(env: dict[str, str]) -> None:
    env |= {
        "AGENT_HUB_CLAUDE_PERMISSION_MODE": "acceptEdits",
        "AGENT_HUB_CLAUDE_MODEL": "claude-opus-5-5",
        "AGENT_HUB_CLAUDE_MAX_BUDGET_USD": "2.50",
        "AGENT_HUB_APPROVAL_TIMEOUT_SECONDS": "30",
    }
    settings = load_settings(env)

    assert settings.claude.permission_mode is PermissionMode.ACCEPT_EDITS
    assert settings.claude.model == "claude-opus-5-5"
    assert settings.claude.max_budget_usd == Decimal("2.50")
    assert settings.approval_timeout_seconds == 30


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENT_HUB_TELEGRAM_TOKEN", ""),
        ("AGENT_HUB_CHAT_ID", "abc"),
        ("AGENT_HUB_ALLOWED_USER_IDS", " , "),
        ("AGENT_HUB_WORKSPACE_ROOT", "/definitely/missing/dir"),
        ("AGENT_HUB_CLAUDE_PERMISSION_MODE", "yolo"),
        ("AGENT_HUB_CLAUDE_MAX_BUDGET_USD", "-1"),
        ("AGENT_HUB_CLAUDE_MAX_BUDGET_USD", "NaN"),
        ("AGENT_HUB_APPROVAL_TIMEOUT_SECONDS", "0"),
    ],
)
def test_invalid_values_are_rejected(env: dict[str, str], name: str, value: str) -> None:
    env[name] = value
    with pytest.raises(ConfigError):
        load_settings(env)
