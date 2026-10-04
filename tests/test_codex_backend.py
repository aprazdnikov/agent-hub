import asyncio
import gc
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from agent_hub.backends.codex import (
    APP_SERVER_FAILED,
    NOT_LOGGED_IN,
    CodexBackend,
    Launcher,
    _child_env,
    _stop,
    app_server,
    converse,
    handshake,
    probe_auth,
)
from agent_hub.backends.codex_protocol import CodexAuth, TurnId, TurnTracker
from agent_hub.backends.rpc import (
    Connection,
    JsonObject,
    Notification,
    RequestHandler,
    RpcError,
    TransportClosedError,
    UnsupportedRequestError,
)
from agent_hub.config import CodexApproval, CodexSandbox, CodexSettings
from agent_hub.domain import (
    AgentEvent,
    Allowed,
    Answered,
    AssistantText,
    BackendKind,
    Failed,
    Finished,
    Prompt,
    SessionId,
    SessionStarted,
    TopicSession,
)
from tests.fakes import FakeChannel

THREAD = SessionId("th-1")
SETTINGS = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, None)
REJECTED = RpcError(-32600, "no active turn")
CHATGPT = {"account": {"type": "chatgpt", "email": None, "planType": "plus"}}


def _completed(turn_id: str, status: str = "completed") -> Notification:
    return Notification(
        "turn/completed", {"threadId": THREAD, "turn": {"id": turn_id, "status": status}}
    )


def _message(text: str) -> Notification:
    item = {"type": "agentMessage", "id": "m", "text": text}
    return Notification("item/completed", {"threadId": THREAD, "turnId": "turn-1", "item": item})


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class FakeThread:
    def __init__(self, steer_error: RpcError | None = None) -> None:
        self.steer_error = steer_error
        self.incoming: asyncio.Queue[Notification | None] = asyncio.Queue()
        self.turns: list[str] = []
        self.steered: list[tuple[TurnId, str]] = []

    async def start_turn(self, prompt: Prompt) -> TurnId:
        self.turns.append(prompt.text)
        return TurnId(f"turn-{len(self.turns)}")

    async def steer(self, turn: TurnId, prompt: Prompt) -> None:
        if self.steer_error is not None:
            raise self.steer_error
        self.steered.append((turn, prompt.text))

    async def notification(self) -> Notification:
        notification = await self.incoming.get()
        if notification is None:
            raise TransportClosedError("closed")
        return notification

    def feed(self, *notifications: Notification | None) -> None:
        for notification in notifications:
            self.incoming.put_nowait(notification)


async def _collect(thread: FakeThread, inbox: asyncio.Queue[Prompt]) -> list[AgentEvent]:
    return [event async for event in converse(thread, TurnTracker(THREAD), Prompt("задача"), inbox)]


async def test_turn_relays_text_and_finishes() -> None:
    thread = FakeThread()
    thread.feed(_message("Готово"), _completed("turn-1"))

    events = await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)

    assert events == [AssistantText("Готово"), Finished(THREAD, None, None)]
    assert thread.turns == ["задача"]


async def test_prompt_during_a_turn_steers_it() -> None:
    thread = FakeThread()
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("ещё"))
    await _settle()
    thread.feed(_completed("turn-1"))

    events = await asyncio.wait_for(run, 1)
    assert thread.steered == [(TurnId("turn-1"), "ещё")]
    assert events == [Finished(THREAD, None, None)]


async def test_rejected_steer__turn_still_running__waits_for_it_to_complete() -> None:
    thread = FakeThread(steer_error=REJECTED)
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("ещё"))
    await _settle()
    assert thread.turns == ["задача"]

    thread.steer_error = None
    thread.feed(_completed("turn-1"))
    await _settle()
    assert thread.turns == ["задача", "ещё"]
    assert not run.done()

    thread.feed(_completed("turn-2"))
    events = await asyncio.wait_for(run, 1)
    assert events == [Finished(THREAD, None, None), Finished(THREAD, None, None)]


async def test_rejected_steers__two_in_a_row__are_delivered_in_order() -> None:
    thread = FakeThread(steer_error=REJECTED)
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("первое"))
    await _settle()
    inbox.put_nowait(Prompt("второе"))
    await _settle()
    assert thread.turns == ["задача"]
    assert thread.steered == []

    thread.steer_error = None
    thread.feed(_completed("turn-1"))
    await _settle()
    assert thread.turns == ["задача", "первое"]
    assert thread.steered == [(TurnId("turn-2"), "второе")]

    thread.feed(_completed("turn-2"))
    events = await asyncio.wait_for(run, 1)
    assert events == [Finished(THREAD, None, None), Finished(THREAD, None, None)]


async def test_rejected_steer__into_the_next_turn__waits_for_that_turn_too() -> None:
    thread = FakeThread(steer_error=REJECTED)
    inbox: asyncio.Queue[Prompt] = asyncio.Queue()
    run = asyncio.create_task(_collect(thread, inbox))
    await _settle()

    inbox.put_nowait(Prompt("первое"))
    await _settle()
    inbox.put_nowait(Prompt("второе"))
    await _settle()

    thread.feed(_completed("turn-1"))
    await _settle()
    assert thread.turns == ["задача", "первое"]
    assert thread.steered == []

    thread.feed(_completed("turn-2"))
    await _settle()
    assert thread.turns == ["задача", "первое", "второе"]
    assert not run.done()

    thread.feed(_completed("turn-3"))
    events = await asyncio.wait_for(run, 1)
    assert events == [Finished(THREAD, None, None)] * 3


async def test_failed_turn_ends_the_session() -> None:
    thread = FakeThread()
    thread.feed(_completed("turn-1", "interrupted"))

    events = await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)

    assert events == [Failed("Ход Codex прерван")]


async def test_closed_transport_propagates() -> None:
    thread = FakeThread()
    thread.feed(None)

    with pytest.raises(TransportClosedError):
        await asyncio.wait_for(_collect(thread, asyncio.Queue()), 1)


type Response = JsonObject | RpcError | TimeoutError | list[JsonObject]


class FakeConnection:
    """Answers each method with its response; a list is answered one item per call."""

    def __init__(self, responses: dict[str, Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, JsonObject]] = []
        self.notified: list[str] = []
        self.incoming: asyncio.Queue[Notification | None] = asyncio.Queue()

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        self.calls.append((method, params))
        response = self.responses[method]
        if isinstance(response, RpcError | TimeoutError):
            raise response
        if isinstance(response, list):
            return response.pop(0)
        return response

    async def notify(self, method: str) -> None:
        self.notified.append(method)

    async def notification(self) -> Notification:
        notification = await self.incoming.get()
        if notification is None:
            raise TransportClosedError("closed")
        return notification


def _launcher(connection: FakeConnection) -> Launcher:
    @asynccontextmanager
    async def launch(_handler: RequestHandler) -> AsyncIterator[Connection]:
        yield connection

    return launch


def _responses(**overrides: Response) -> dict[str, Response]:
    return {
        "initialize": {},
        "account/read": CHATGPT,
        "thread/start": {"thread": {"id": "th-1"}},
        "thread/resume": {"thread": {"id": "th-1"}},
        "turn/start": {"turn": {"id": "turn-1"}},
        **{key.replace("_", "/"): value for key, value in overrides.items()},
    }


async def _run(backend: CodexBackend, session: TopicSession) -> list[AgentEvent]:
    channel = FakeChannel(Allowed(), Answered(()))
    events = backend.run(session, Prompt("задача"), channel, asyncio.Queue())
    return [event async for event in events]


async def test_new_session_starts_a_thread_and_runs_a_turn(tmp_path: Path) -> None:
    connection = FakeConnection(_responses())
    connection.incoming.put_nowait(_completed("turn-1"))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await asyncio.wait_for(
        _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None)), 1
    )

    assert events == [SessionStarted(THREAD), Finished(THREAD, None, None)]
    assert [method for method, _ in connection.calls] == [
        "initialize",
        "account/read",
        "thread/start",
        "turn/start",
    ]
    assert connection.notified == ["initialized"]


async def test_run__closed_early__stops_waiting_for_codex_at_once(tmp_path: Path) -> None:
    connection = FakeConnection(_responses())
    connection.incoming.put_nowait(_message("Думаю"))
    backend = CodexBackend(SETTINGS, _launcher(connection))
    channel = FakeChannel(Allowed(), Answered(()))
    session = TopicSession(BackendKind.CODEX, tmp_path, None)
    events = backend.run(session, Prompt("задача"), channel, asyncio.Queue())

    assert [await anext(events), await anext(events)] == [
        SessionStarted(THREAD),
        AssistantText("Думаю"),
    ]
    await events.aclose()

    waiting = asyncio.all_tasks() - {asyncio.current_task()}
    assert all(task.done() or task.cancelling() for task in waiting)


async def test_run__rpc_error_in_the_session__fails_with_its_message(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(turn_start=RpcError(-32000, "model unavailable")))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [SessionStarted(THREAD), Failed("Codex: model unavailable")]


async def test_run__request_timeout_in_the_session__fails_with_app_server_failed(
    tmp_path: Path,
) -> None:
    connection = FakeConnection(_responses(turn_start=TimeoutError()))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [SessionStarted(THREAD), Failed(APP_SERVER_FAILED)]


async def test_run__transport_closed_mid_turn__fails_with_app_server_failed(
    tmp_path: Path,
) -> None:
    connection = FakeConnection(_responses())
    connection.incoming.put_nowait(None)
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [SessionStarted(THREAD), Failed(APP_SERVER_FAILED)]


async def test_run__cancelled_during_a_turn__propagates_and_stops_the_app_server(
    tmp_path: Path,
) -> None:
    connection = FakeConnection(_responses())
    exited = asyncio.Event()

    @asynccontextmanager
    async def launch(_handler: RequestHandler) -> AsyncIterator[Connection]:
        try:
            yield connection
        finally:
            exited.set()

    backend = CodexBackend(SETTINGS, launch)
    running = asyncio.create_task(_run(backend, TopicSession(BackendKind.CODEX, tmp_path, None)))
    await _settle()
    assert "turn/start" in [method for method, _ in connection.calls]

    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 1)
    assert exited.is_set()


async def test_missing_login_fails_before_opening_a_thread(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(account_read=MISSING_ACCOUNT))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [Failed(NOT_LOGGED_IN)]


async def test_resume_failure_suggests_reset(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(thread_resume=RpcError(-32600, "no rollout found")))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, THREAD))

    assert len(events) == 1
    assert isinstance(events[0], Failed)
    assert "no rollout found" in events[0].reason
    assert "/reset" in events[0].reason


class FailedSpawn:
    async def __aenter__(self) -> Connection:
        raise FileNotFoundError("codex")

    async def __aexit__(self, *_exc: object) -> None:
        return None


async def test_spawn_failure_fails_the_turn(tmp_path: Path) -> None:
    def broken(_handler: RequestHandler) -> FailedSpawn:
        return FailedSpawn()

    events = await _run(
        CodexBackend(SETTINGS, broken), TopicSession(BackendKind.CODEX, tmp_path, None)
    )

    assert events == [Failed(APP_SERVER_FAILED)]


KEYED = CodexSettings(None, CodexSandbox.WORKSPACE_WRITE, CodexApproval.ON_REQUEST, "sk-test")
API_KEY_ACCOUNT = {"account": {"type": "apiKey"}}
MISSING_ACCOUNT = {"account": None, "requiresOpenaiAuth": True}
KEY_LOGIN = ("account/login/start", {"type": "apiKey", "apiKey": "sk-test"})


async def test_probe_auth__no_login_and_a_key__logs_in_with_the_key() -> None:
    connection = FakeConnection(
        _responses(
            account_read=[MISSING_ACCOUNT, API_KEY_ACCOUNT], account_login_start={"type": "apiKey"}
        )
    )

    assert await probe_auth(KEYED, _launcher(connection)) is CodexAuth.API_KEY
    assert KEY_LOGIN in connection.calls


async def test_probe_auth__api_key_login_and_a_key__logs_in_again_to_rotate_the_key() -> None:
    connection = FakeConnection(
        _responses(
            account_read=[API_KEY_ACCOUNT, API_KEY_ACCOUNT], account_login_start={"type": "apiKey"}
        )
    )

    assert await probe_auth(KEYED, _launcher(connection)) is CodexAuth.API_KEY
    assert KEY_LOGIN in connection.calls


async def test_probe_auth__chatgpt_login_and_a_key__keeps_the_chatgpt_login() -> None:
    connection = FakeConnection(_responses())

    assert await probe_auth(KEYED, _launcher(connection)) is CodexAuth.CHATGPT
    assert [method for method, _ in connection.calls] == ["initialize", "account/read"]


async def test_run__no_login_and_a_key__logs_in_with_the_key_and_starts(tmp_path: Path) -> None:
    connection = FakeConnection(
        _responses(
            account_read=[MISSING_ACCOUNT, API_KEY_ACCOUNT], account_login_start={"type": "apiKey"}
        )
    )
    backend = CodexBackend(KEYED, _launcher(connection))
    connection.incoming.put_nowait(_completed("turn-1"))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [SessionStarted(THREAD), Finished(THREAD, None, None)]
    assert KEY_LOGIN in connection.calls


async def test_run__no_login_and_no_key__fails_without_logging_in(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(account_read=MISSING_ACCOUNT))
    backend = CodexBackend(SETTINGS, _launcher(connection))

    events = await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert events == [Failed(NOT_LOGGED_IN)]
    assert "account/login/start" not in [method for method, _ in connection.calls]


async def test_run__api_key_login_and_a_key__does_not_log_in_again(tmp_path: Path) -> None:
    connection = FakeConnection(_responses(account_read=API_KEY_ACCOUNT))
    backend = CodexBackend(KEYED, _launcher(connection))
    connection.incoming.put_nowait(_completed("turn-1"))

    await _run(backend, TopicSession(BackendKind.CODEX, tmp_path, None))

    assert "account/login/start" not in [method for method, _ in connection.calls]


async def test_real_app_server_reports_missing_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    assert await asyncio.wait_for(probe_auth(SETTINGS), 60) is CodexAuth.MISSING


async def test_missing_login_fails_with_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    session = TopicSession(BackendKind.CODEX, tmp_path, None)

    events = await asyncio.wait_for(_run(CodexBackend(SETTINGS), session), 60)

    assert events == [Failed(NOT_LOGGED_IN)]


async def test_real_app_server__cancelled_after_the_handshake__stops_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    ready = asyncio.Event()

    async def refuse(method: str, _params: JsonObject) -> JsonObject:
        raise UnsupportedRequestError(method)

    async def session() -> None:
        async with app_server(refuse) as connection:
            await handshake(connection)
            ready.set()
            await asyncio.Event().wait()

    running = asyncio.create_task(session())
    await asyncio.wait_for(ready.wait(), 60)

    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 30)
    gc.collect()


async def test_real_app_server_accepts_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert await asyncio.wait_for(probe_auth(KEYED), 60) is CodexAuth.API_KEY


def test_child_env__secrets_in_the_hub_env__are_not_passed_to_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_HUB_TELEGRAM_TOKEN", "123:secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AGENT_HUB_WORKSPACE_ROOT", "/w")

    env = _child_env()

    assert "AGENT_HUB_TELEGRAM_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["AGENT_HUB_WORKSPACE_ROOT"] == "/w"


async def _child(code: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, stdin=asyncio.subprocess.PIPE
    )


READS_STDIN = "import sys; sys.stdin.read()"
IGNORES_STDIN = "import time; time.sleep(60)"


async def test_stop__child_exits_on_stdin_eof__is_not_killed() -> None:
    process = await _child(READS_STDIN)

    await asyncio.wait_for(_stop(process, 30), 30)

    assert process.returncode == 0


async def test_stop__child_ignores_stdin_eof__is_killed_after_the_grace() -> None:
    process = await _child(IGNORES_STDIN)

    await asyncio.wait_for(_stop(process, 0.1), 30)

    assert process.returncode not in (None, 0)


async def test_stop__cancelled_during_the_grace__still_kills_the_child() -> None:
    process = await _child(IGNORES_STDIN)
    stopping = asyncio.create_task(_stop(process, 60))
    await _settle()

    stopping.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopping, 30)
    assert process.returncode not in (None, 0)
