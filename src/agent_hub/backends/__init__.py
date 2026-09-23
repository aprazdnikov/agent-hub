"""Agent backends: one adapter per AI agent, all behind `AgentBackend`."""

from collections.abc import AsyncGenerator
from typing import Protocol

from agent_hub.domain import AgentEvent, Decision, ToolRequest, TopicSession


class ApprovalGate(Protocol):
    """Asks the human whether a tool call may run."""

    async def request(self, tool: ToolRequest) -> Decision: ...


class AgentBackend(Protocol):
    def run(
        self, session: TopicSession, prompt: str, gate: ApprovalGate
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run one user turn.

        Must end with exactly one `Finished` or `Failed` event and must not raise for
        agent-side failures; cancellation is propagated as `CancelledError`.
        """
        ...
