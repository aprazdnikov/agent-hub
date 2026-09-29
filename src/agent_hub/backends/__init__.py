"""Agent backends: one adapter per AI agent, all behind `AgentBackend`."""

from collections.abc import AsyncGenerator, Sequence
from typing import Protocol

from agent_hub.domain import (
    AgentEvent,
    Decision,
    FileDelivery,
    OutgoingFile,
    Prompt,
    Question,
    QuestionsOutcome,
    ToolRequest,
    TopicSession,
)


class UserChannel(Protocol):
    """The human on the other side: approves tools, answers questions, receives files."""

    async def request(self, tool: ToolRequest) -> Decision: ...

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome: ...

    async def send_file(self, file: OutgoingFile) -> FileDelivery: ...


class Inbox(Protocol):
    """Further user messages for a session that is still open."""

    async def get(self) -> Prompt: ...


class AgentBackend(Protocol):
    def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel, inbox: Inbox
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run an agent session starting with `prompt`.

        The session stays open while the agent is busy or has background tasks, taking
        further prompts from `inbox`. Every agent turn ends with one `Finished` or
        `Failed` event. Agent-side failures are events, not exceptions; cancellation is
        propagated as `CancelledError`.
        """
        ...
