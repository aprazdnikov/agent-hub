import logging
from pathlib import Path

import pytest

from agent_hub import __main__ as entry
from agent_hub.backends.codex_protocol import CodexAuth
from agent_hub.config import Settings, load_settings
from agent_hub.domain import BackendKind, TopicKey, TopicSession
from agent_hub.store import TopicStore


def settings_for(tmp_path: Path, **extra: str) -> Settings:
    return load_settings(
        {
            "AGENT_HUB_TELEGRAM_TOKEN": "123:abc",
            "AGENT_HUB_CHAT_ID": "-1001234567890",
            "AGENT_HUB_ALLOWED_USER_IDS": "111",
            "AGENT_HUB_WORKSPACE_ROOT": str(tmp_path),
            **extra,
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> TopicStore:
    return TopicStore.open(tmp_path / "topics.json")


def test_codex_unused_by_default(tmp_path: Path, store: TopicStore) -> None:
    assert not entry.codex_in_use(settings_for(tmp_path), store)


def test_codex_in_use_as_default_backend(tmp_path: Path, store: TopicStore) -> None:
    settings = settings_for(tmp_path, AGENT_HUB_DEFAULT_BACKEND="codex")
    assert entry.codex_in_use(settings, store)


def test_codex_in_use_with_api_key(tmp_path: Path, store: TopicStore) -> None:
    settings = settings_for(tmp_path, OPENAI_API_KEY="sk-test")
    assert entry.codex_in_use(settings, store)


def test_codex_in_use_with_codex_topic(tmp_path: Path, store: TopicStore) -> None:
    store.put(TopicKey(1, 2), TopicSession(BackendKind.CODEX, tmp_path, None))
    assert entry.codex_in_use(settings_for(tmp_path), store)


async def test_check_codex_logs_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing(_settings: object) -> CodexAuth:
        raise OSError

    monkeypatch.setattr(entry, "probe_auth", failing)
    with caplog.at_level(logging.WARNING, logger="agent_hub"):
        await entry.check_codex(settings_for(tmp_path).codex)
    assert "codex unavailable" in caplog.text


async def test_check_codex_logs_when_not_logged_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def missing(_settings: object) -> CodexAuth:
        return CodexAuth.MISSING

    monkeypatch.setattr(entry, "probe_auth", missing)
    with caplog.at_level(logging.WARNING, logger="agent_hub"):
        await entry.check_codex(settings_for(tmp_path).codex)
    assert "codex not logged in" in caplog.text
