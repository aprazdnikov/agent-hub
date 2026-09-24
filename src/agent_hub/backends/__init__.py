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


class AgentBackend(Protocol):
    def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one user turn.

        Must end with exactly one `Finished` or `Failed` event and must not raise for
        agent-side failures; cancellation is propagated as `CancelledError`.
        """
        ...
