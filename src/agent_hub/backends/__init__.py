"""Agent backends: one adapter per AI agent, all behind `AgentBackend`."""

from collections.abc import AsyncGenerator, Sequence
from typing import Protocol

from agent_hub.domain import (
    AgentEvent,
    Decision,
    Prompt,
    Question,
    QuestionsOutcome,
    ToolRequest,
    TopicSession,
)


class ApprovalGate(Protocol):
    """Asks the human whether a tool call may run, or to answer the agent's questions."""

    async def request(self, tool: ToolRequest) -> Decision: ...

    async def ask(self, questions: Sequence[Question]) -> QuestionsOutcome: ...


class AgentBackend(Protocol):
    def run(
        self, session: TopicSession, prompt: Prompt, gate: ApprovalGate
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one user turn.

        Must end with exactly one `Finished` or `Failed` event and must not raise for
        agent-side failures; cancellation is propagated as `CancelledError`.
        """
        ...
