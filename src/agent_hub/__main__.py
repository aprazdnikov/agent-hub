"""Entry point: `uv run agent-hub`."""

import asyncio
import logging
import os
import sys
from typing import assert_never

from telegram import Update

from agent_hub.backends import AgentBackend
from agent_hub.backends.claude import ClaudeBackend
from agent_hub.backends.codex import CodexBackend, probe_auth
from agent_hub.backends.codex_protocol import CodexAuth
from agent_hub.backends.rpc import ProtocolError, RpcError, TransportClosedError
from agent_hub.bot import Hub
from agent_hub.config import CodexSettings, ConfigError, Settings, load_settings
from agent_hub.domain import BackendKind
from agent_hub.logs import configure_logging
from agent_hub.store import CorruptStateError, TopicStore

log = logging.getLogger("agent_hub")

PROBE_TIMEOUT_SECONDS = 30


def build_backend(kind: BackendKind, settings: Settings) -> AgentBackend:
    match kind:
        case BackendKind.CLAUDE:
            return ClaudeBackend(settings.claude, settings.background_timeout_seconds)
        case BackendKind.CODEX:
            return CodexBackend(settings.codex)
        case _:
            assert_never(kind)


def codex_in_use(settings: Settings, store: TopicStore) -> bool:
    return (
        settings.default_backend is BackendKind.CODEX
        or settings.codex.api_key is not None
        or store.uses(BackendKind.CODEX)
    )


async def check_codex(settings: CodexSettings) -> None:
    """Report whether Codex topics can work; never stops the hub."""
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            auth = await probe_auth(settings)
    except (OSError, RpcError, TransportClosedError, ProtocolError, TimeoutError):
        log.warning("codex unavailable", exc_info=True)
        return
    match auth:
        case CodexAuth.MISSING:
            log.warning("codex not logged in")
        case CodexAuth.API_KEY | CodexAuth.CHATGPT | CodexAuth.OTHER | CodexAuth.NOT_REQUIRED:
            log.info("codex ready", extra={"auth": auth.value})
        case _:
            assert_never(auth)


def main() -> None:
    configure_logging()
    try:
        settings = load_settings(os.environ)
        store = TopicStore.open(settings.state_file)
    except (ConfigError, CorruptStateError) as error:
        sys.exit(f"agent-hub: {error}")
    backends = {kind: build_backend(kind, settings) for kind in BackendKind}
    startup = [lambda: check_codex(settings.codex)] if codex_in_use(settings, store) else []
    log.info(
        "starting",
        extra={
            "chat_id": settings.chat_id,
            "workspace_root": str(settings.workspace_root),
            "permission_mode": settings.claude.permission_mode.value,
            "default_backend": settings.default_backend.value,
        },
    )
    Hub(settings, store, backends, startup=startup).build_application().run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
