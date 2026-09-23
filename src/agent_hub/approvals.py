"""Pending human decisions on tool calls, keyed by an opaque id carried in button data."""

import asyncio
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, assert_never, final

from agent_hub.domain import Allowed, Decision, Denied

ApprovalId = NewType("ApprovalId", str)

CALLBACK_PREFIX = "ap"
_CALLBACK_PARTS = 3  # prefix:verdict:id
DENIED_BY_USER = "Пользователь запретил этот вызов"


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


@final
@dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    approval_id: ApprovalId
    verdict: Verdict

    @property
    def decision(self) -> Decision:
        match self.verdict:
            case Verdict.ALLOW:
                return Allowed()
            case Verdict.DENY:
                return Denied(DENIED_BY_USER)
            case _:
                assert_never(self.verdict)


class ApprovalRegistry:
    def __init__(self) -> None:
        self._pending: dict[ApprovalId, asyncio.Future[Decision]] = {}

    def open(self) -> tuple[ApprovalId, asyncio.Future[Decision]]:
        approval_id = ApprovalId(secrets.token_hex(8))
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = future
        return approval_id, future

    def close(self, approval_id: ApprovalId) -> None:
        self._pending.pop(approval_id, None)

    def resolve(self, answer: ApprovalAnswer) -> bool:
        """Deliver a decision; False when the request is gone (answered, timed out, stopped)."""
        future = self._pending.pop(answer.approval_id, None)
        if future is None or future.done():
            return False
        future.set_result(answer.decision)
        return True


def callback_data(approval_id: ApprovalId, verdict: Verdict) -> str:
    return f"{CALLBACK_PREFIX}:{verdict.value}:{approval_id}"


def parse_callback_data(data: str) -> ApprovalAnswer | None:
    parts = data.split(":")
    if len(parts) != _CALLBACK_PARTS or parts[0] != CALLBACK_PREFIX or not parts[2]:
        return None
    try:
        verdict = Verdict(parts[1])
    except ValueError:
        return None
    return ApprovalAnswer(ApprovalId(parts[2]), verdict)
