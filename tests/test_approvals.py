import asyncio

import pytest

from agent_hub.approvals import (
    ApprovalAnswer,
    ApprovalId,
    ApprovalRegistry,
    Verdict,
    callback_data,
    parse_callback_data,
)
from agent_hub.domain import Allowed, Denied


@pytest.mark.parametrize("verdict", list(Verdict))
def test_callback_data_round_trips(verdict: Verdict) -> None:
    approval_id = ApprovalId("0123456789abcdef")
    data = callback_data(approval_id, verdict)
    assert len(data.encode()) <= 64  # Telegram callback_data limit
    assert parse_callback_data(data) == ApprovalAnswer(approval_id, verdict)


@pytest.mark.parametrize(
    "data", ["", "ap", "ap:allow:", "ap:maybe:x", "xx:allow:x", "ap:allow:x:y"]
)
def test_malformed_callback_data_is_rejected(data: str) -> None:
    assert parse_callback_data(data) is None


async def test_resolve_delivers_decision_once() -> None:
    registry = ApprovalRegistry()
    approval_id, future = registry.open()

    assert registry.resolve(ApprovalAnswer(approval_id, Verdict.ALLOW)) is True
    assert await future == Allowed()
    assert registry.resolve(ApprovalAnswer(approval_id, Verdict.DENY)) is False


async def test_closed_request_cannot_be_resolved() -> None:
    registry = ApprovalRegistry()
    approval_id, future = registry.open()
    registry.close(approval_id)

    assert registry.resolve(ApprovalAnswer(approval_id, Verdict.ALLOW)) is False
    assert not future.done()


async def test_deny_carries_reason() -> None:
    registry = ApprovalRegistry()
    approval_id, future = registry.open()
    registry.resolve(ApprovalAnswer(approval_id, Verdict.DENY))

    decision = await asyncio.wait_for(future, 1)
    assert isinstance(decision, Denied)
    assert decision.reason
