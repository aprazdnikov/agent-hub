import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    UserMessage,
)
from claude_agent_sdk.types import MessageOrigin

from agent_hub.backends.claude import ClaudeBackend, Phase, SessionActivity
from agent_hub.config import ClaudeSettings, PermissionMode
from agent_hub.domain import AgentEvent, AssistantText, BackgroundAbandoned, Finished, Prompt


def _started(
    task_id: str, description: str = "sleep", *, background: bool = True
) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={"is_backgrounded": background},
        task_id=task_id,
        description=description,
        uuid="u",
        session_id="s",
    )


def _notified(task_id: str) -> TaskNotificationMessage:
    return TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id=task_id,
        status="completed",
        output_file="out.log",
        summary="done",
        uuid="u",
        session_id="s",
    )


def _completed(task_id: str) -> TaskUpdatedMessage:
    return TaskUpdatedMessage(
        subtype="task_updated", data={}, task_id=task_id, patch={"status": "completed"}
    )


def _result(origin: MessageOrigin | None = None) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        origin=origin,
    )


INJECTED: MessageOrigin = {"kind": "task-notification"}
TEXT = AssistantMessage(content=[TextBlock(text="BG-DONE")], model="claude")


def _accepted(prompt_id: str) -> UserMessage:
    """The CLI echoing a prompt back once it takes it into a turn."""
    return UserMessage(content="…", uuid=prompt_id)


def _phase(activity: SessionActivity) -> Phase:
    # A call, not an attribute: mypy would otherwise keep the narrowed type between steps.
    return activity.phase


def _activity(*messages: Message) -> SessionActivity:
    activity = SessionActivity()
    activity.sent("p1")
    activity.observe(_accepted("p1"))
    for message in messages:
        activity.observe(message)
    return activity


def test_plain_turn_is_busy_until_its_result() -> None:
    activity = _activity()
    assert _phase(activity) is Phase.BUSY

    activity.observe(_result())
    assert _phase(activity) is Phase.IDLE


def test_foreground_subagent_does_not_keep_session_open() -> None:
    activity = _activity(
        _started("a", background=False), _completed("a"), _notified("a"), TEXT, _result()
    )
    assert _phase(activity) is Phase.IDLE


def test_background_task_keeps_session_open_until_injected_turn_ends() -> None:
    activity = _activity(_started("b", "sleep 8"), _result())
    assert _phase(activity) is Phase.BACKGROUND
    assert activity.background == ("sleep 8",)

    activity.observe(_notified("b"))
    assert _phase(activity) is Phase.SETTLING

    activity.observe(SystemMessage(subtype="init", data={}))
    assert _phase(activity) is Phase.BUSY

    activity.observe(TEXT)
    activity.observe(_result(INJECTED))
    assert _phase(activity) is Phase.IDLE


def test_task_update_before_notification_waits_for_the_injected_turn() -> None:
    activity = _activity(_started("b"), _result(), _completed("b"))
    assert _phase(activity) is Phase.SETTLING

    activity.observe(_notified("b"))
    activity.observe(TEXT)
    activity.observe(_result(INJECTED))
    assert _phase(activity) is Phase.IDLE


def test_background_task_finishing_during_a_turn_still_expects_its_report() -> None:
    activity = _activity(_started("b"), _completed("b"), _result())
    assert _phase(activity) is Phase.SETTLING


def test_task_ending_during_an_injected_turn_expects_another_one() -> None:
    activity = _activity(_started("b1"), _started("b2"), _result(), _notified("b1"), TEXT)
    activity.observe(_notified("b2"))
    activity.observe(_result(INJECTED))
    assert _phase(activity) is Phase.SETTLING


def test_one_injected_turn_may_report_several_tasks() -> None:
    activity = _activity(_started("b1"), _started("b2"), _result(), _notified("b1"))
    assert _phase(activity) is Phase.BACKGROUND

    activity.observe(_notified("b2"))
    activity.observe(TEXT)
    activity.observe(_result(INJECTED))
    assert _phase(activity) is Phase.IDLE


def test_message_sent_while_background_runs_is_busy_until_its_result() -> None:
    activity = _activity(_started("b"), _result())
    activity.sent("p2")
    assert _phase(activity) is Phase.BUSY

    activity.observe(_accepted("p2"))
    activity.observe(_result())
    assert _phase(activity) is Phase.BACKGROUND


def test_message_merged_into_the_running_turn_ends_with_it() -> None:
    activity = _activity()
    activity.sent("p2")
    activity.observe(_accepted("p2"))

    activity.observe(_result())
    assert _phase(activity) is Phase.IDLE


def test_message_not_yet_accepted_keeps_session_busy_after_a_result() -> None:
    activity = _activity()
    activity.sent("p2")

    activity.observe(_result())
    assert _phase(activity) is Phase.BUSY


def test_injected_result_does_not_answer_a_human_prompt() -> None:
    activity = _activity(_started("b"), _result(), _notified("b"))
    activity.sent("p2")
    activity.observe(_result(INJECTED))
    assert _phase(activity) is Phase.BUSY


def test_background_count_is_reported_with_result() -> None:
    activity = _activity(_started("b"))
    assert activity.background == ("sleep",)


class FakeConversation:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Message | None] = asyncio.Queue()
        self.sent: list[str] = []
        self.queried = asyncio.Event()

    async def receive_messages(self) -> AsyncIterator[Message]:
        while (message := await self.incoming.get()) is not None:
            yield message

    async def query(self, prompt: str | AsyncIterable[dict[str, Any]]) -> None:
        assert not isinstance(prompt, str)
        async for message in prompt:
            self.sent.append(message["message"]["content"][-1]["text"])
            self.incoming.put_nowait(_accepted(message["uuid"]))
        self.queried.set()

    def feed(self, *messages: Message | None) -> None:
        for message in messages:
            self.incoming.put_nowait(message)


def _backend(background_timeout: int = 60) -> ClaudeBackend:
    return ClaudeBackend(ClaudeSettings(PermissionMode.DEFAULT, None, None, background_timeout))


async def _converse(
    conversation: FakeConversation, inbox: asyncio.Queue[Prompt], backend: ClaudeBackend
) -> list[AgentEvent]:
    activity = SessionActivity()
    activity.sent("p1")
    activity.observe(_accepted("p1"))
    return [event async for event in backend.converse(conversation, inbox, activity)]


async def test_session_stays_open_for_background_result() -> None:
    conversation = FakeConversation()
    conversation.feed(_started("b"), _result(), _notified("b"), TEXT, _result(INJECTED))

    events = await asyncio.wait_for(_converse(conversation, asyncio.Queue(), _backend()), 1)

    finished = [event for event in events if isinstance(event, Finished)]
    assert [event.background for event in finished] == [1, 0]
    assert AssistantText("BG-DONE") in events


async def test_prompt_during_background_is_sent_to_the_same_session() -> None:
    conversation = FakeConversation()
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    conversation.feed(_started("b"), _result())
    run = asyncio.create_task(_converse(conversation, inbox, _backend()))
    await asyncio.sleep(0)

    inbox.put_nowait(Prompt("ещё вопрос"))
    await asyncio.wait_for(conversation.queried.wait(), 1)
    conversation.feed(_result(), _completed("b"), _notified("b"), TEXT, _result(INJECTED))

    await asyncio.wait_for(run, 1)
    assert conversation.sent == ["ещё вопрос"]


async def test_background_outliving_the_budget_is_abandoned() -> None:
    conversation = FakeConversation()
    conversation.feed(_started("b", "sleep 600"), _result())

    events = await asyncio.wait_for(_converse(conversation, asyncio.Queue(), _backend(0)), 1)

    assert events[-1] == BackgroundAbandoned(("sleep 600",))


async def test_cli_exit_ends_the_session() -> None:
    conversation = FakeConversation()
    conversation.feed(_started("b"), _result(), None)

    events = await asyncio.wait_for(_converse(conversation, asyncio.Queue(), _backend()), 1)

    assert isinstance(events[-1], Finished)
