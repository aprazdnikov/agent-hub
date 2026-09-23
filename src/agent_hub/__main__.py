"""Entry point: `uv run agent-hub`."""

import logging
import os
import sys
from typing import assert_never

from telegram import Update

from agent_hub.backends import AgentBackend
from agent_hub.backends.claude import ClaudeBackend
from agent_hub.bot import Hub
from agent_hub.config import ConfigError, Settings, load_settings
from agent_hub.domain import BackendKind
from agent_hub.logs import configure_logging
from agent_hub.store import CorruptStateError, TopicStore

log = logging.getLogger("agent_hub")


def build_backend(kind: BackendKind, settings: Settings) -> AgentBackend:
    match kind:
        case BackendKind.CLAUDE:
            return ClaudeBackend(settings.claude)
        case _:
            assert_never(kind)


def main() -> None:
    configure_logging()
    try:
        settings = load_settings(os.environ)
        store = TopicStore.open(settings.state_file)
    except (ConfigError, CorruptStateError) as error:
        sys.exit(f"agent-hub: {error}")
    backends = {kind: build_backend(kind, settings) for kind in BackendKind}
    log.info(
        "starting",
        extra={
            "chat_id": settings.chat_id,
            "workspace_root": str(settings.workspace_root),
            "permission_mode": settings.claude.permission_mode.value,
        },
    )
    Hub(settings, store, backends).build_application().run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
