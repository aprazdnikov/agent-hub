"""OpenAI Codex backend via `codex app-server`, JSON-RPC over the child's stdio."""

import asyncio
import contextlib
import logging
import os
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol, assert_never, final

import codex_cli_bin

from agent_hub.backends import Inbox, UserChannel
from agent_hub.backends.codex_protocol import (
    Call,
    CodexAuth,
    TurnId,
    TurnTracker,
    api_key_login_params,
    auth_state,
    initialize_params,
    open_thread,
    parse_thread_id,
    parse_turn_id,
    steer_params,
    turn_params,
)
from agent_hub.backends.codex_requests import answer
from agent_hub.backends.rpc import (
    Connection,
    JsonObject,
    Notification,
    ProtocolError,
    RequestHandler,
    RpcConnection,
    RpcError,
    TransportClosedError,
    UnsupportedRequestError,
)
from agent_hub.config import OPENAI_API_KEY, PREFIX, CodexSettings
from agent_hub.domain import AgentEvent, Failed, Prompt, SessionId, SessionStarted, TopicSession

log = logging.getLogger(__name__)

# A whole turn item (a diff, a command's output) arrives as one stdout line.
LINE_LIMIT = 64 * 2**20
# Handshake and thread/turn control only; a turn itself runs until it ends or /stop.
REQUEST_TIMEOUT_SECONDS = 60
# app-server exits on stdin EOF; the grace lets it shut down cleanly before a kill.
SHUTDOWN_GRACE_SECONDS = 5
NOT_LOGGED_IN = (
    "Codex не авторизован: выполните `codex login --device-auth` или задайте OPENAI_API_KEY "
    "и перезапустите бота"
)
# Commands Codex runs inherit its environment; app-server takes the key via login, not env.
HIDDEN_FROM_CHILD = frozenset({f"{PREFIX}TELEGRAM_TOKEN", OPENAI_API_KEY})
APP_SERVER_FAILED = "Codex app-server завершился, подробности в логе сервиса"

Launcher = Callable[[RequestHandler], AbstractAsyncContextManager[Connection]]


@asynccontextmanager
async def app_server(handler: RequestHandler) -> AsyncIterator[Connection]:
    """A `codex app-server` child answering server requests with `handler`; stopped on exit."""
    process = await asyncio.create_subprocess_exec(
        codex_cli_bin.bundled_codex_path(),
        "app-server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_child_env(),
        limit=LINE_LIMIT,
    )
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr
    if stdin is None or stdout is None or stderr is None:
        process.kill()
        await process.wait()
        raise TransportClosedError("app-server started without pipes")
    connection = RpcConnection(stdout, stdin, handler)
    tasks = [asyncio.create_task(connection.serve()), asyncio.create_task(_log_stderr(stderr))]
    try:
        yield connection
    finally:
        # The readers keep draining the pipes while the child shuts down, so it cannot block.
        await _stop(process, SHUTDOWN_GRACE_SECONDS)
        for task in tasks:
            task.cancel()
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                log.warning("codex transport failed", exc_info=result)


async def _stop(process: asyncio.subprocess.Process, grace: float) -> None:
    """Close the child's stdin and let it exit within `grace` seconds, then kill it."""
    if process.stdin is not None:
        process.stdin.close()
    try:
        await asyncio.wait_for(process.wait(), grace)
    except TimeoutError:
        log.warning("codex app-server did not exit on stdin close, killing it")
    finally:
        # Also reached when cancelled during the grace wait: the child must not outlive us.
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()


def _child_env() -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name not in HIDDEN_FROM_CHILD}
    tools = codex_cli_bin.bundled_path_dir()
    if tools is not None:
        # Bundled helpers such as ripgrep, put on PATH the way the official SDK does.
        env["PATH"] = os.pathsep.join([str(tools), env.get("PATH", "")])
    return env


async def _log_stderr(stream: asyncio.StreamReader) -> None:
    while line := await stream.readline():
        log.info("codex stderr", extra={"line": line.decode(errors="replace").rstrip()})


async def _call(connection: Connection, method: Call, params: JsonObject) -> JsonObject:
    async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
        return await connection.request(method, params)


async def handshake(connection: Connection) -> None:
    await _call(connection, Call.INITIALIZE, initialize_params())
    await connection.notify(Call.INITIALIZED)


class CodexThread(Protocol):
    """The part of an app-server thread a session needs."""

    async def start_turn(self, prompt: Prompt) -> TurnId: ...

    async def steer(self, turn: TurnId, prompt: Prompt) -> None: ...

    async def notification(self) -> Notification: ...


@final
class RpcThread:
    def __init__(self, connection: Connection, thread_id: SessionId) -> None:
        self._connection = connection
        self._thread_id = thread_id

    async def start_turn(self, prompt: Prompt) -> TurnId:
        params = turn_params(self._thread_id, prompt)
        return parse_turn_id(await _call(self._connection, Call.TURN_START, params))

    async def steer(self, turn: TurnId, prompt: Prompt) -> None:
        params = steer_params(self._thread_id, turn, prompt)
        await _call(self._connection, Call.TURN_STEER, params)

    async def notification(self) -> Notification:
        return await self._connection.notification()


class CodexBackend:
    def __init__(self, settings: CodexSettings, launch: Launcher = app_server) -> None:
        self._settings = settings
        self._launch = launch

    async def run(
        self, session: TopicSession, prompt: Prompt, channel: UserChannel, inbox: Inbox
    ) -> AsyncGenerator[AgentEvent, None]:
        async def handle(method: str, params: JsonObject) -> JsonObject:
            return await answer(channel, method, params)

        try:
            async with self._launch(handle) as connection:
                await handshake(connection)
                auth = auth_state(await _call(connection, Call.ACCOUNT_READ, {}))
                if auth is CodexAuth.MISSING:
                    auth = await ensure_login(connection, self._settings, CodexAuth.MISSING)
                if auth is CodexAuth.MISSING:
                    yield Failed(NOT_LOGGED_IN)
                    return
                method, params = open_thread(session, self._settings)
                try:
                    thread_id = parse_thread_id(await _call(connection, method, params))
                except RpcError as error:
                    yield Failed(_open_failure(session, error))
                    return
                yield SessionStarted(thread_id)
                thread = RpcThread(connection, thread_id)
                conversation = converse(thread, TurnTracker(thread_id), prompt, inbox)
                async with contextlib.aclosing(conversation) as events:
                    async for event in events:
                        yield event
        except RpcError as error:
            yield Failed(f"Codex: {error.message}")
        except (TransportClosedError, ProtocolError, TimeoutError, OSError):
            log.warning("codex session broke", exc_info=True)
            yield Failed(APP_SERVER_FAILED)


def _open_failure(session: TopicSession, error: RpcError) -> str:
    if session.session_id is None:
        return f"Codex не начал сессию: {error.message}"
    return (
        f"Codex не смог продолжить сессию {session.session_id}: {error.message}. "
        "/reset — начать заново"
    )


async def converse(
    thread: CodexThread, tracker: TurnTracker, prompt: Prompt, inbox: Inbox
) -> AsyncGenerator[AgentEvent, None]:
    """Relay a thread until no turn is running and nothing is queued.

    Prompts sent meanwhile join the running turn. One the turn rejects waits, in order with
    the others, for that turn to complete and then starts the next turn; a turn is never
    started while another runs.
    """
    queued: deque[Prompt] = deque()
    active: TurnId | None = await thread.start_turn(prompt)
    next_notification = asyncio.ensure_future(thread.notification())
    next_prompt = asyncio.ensure_future(inbox.get())
    try:
        # A prompt already taken from the inbox must be delivered even if no turn runs.
        while active is not None or queued or next_prompt.done():
            done, _ = await asyncio.wait(
                {next_notification, next_prompt}, return_when=asyncio.FIRST_COMPLETED
            )
            if next_prompt in done:
                queued.append(next_prompt.result())
                active = await _deliver(thread, active, queued)
                next_prompt = asyncio.ensure_future(inbox.get())
            if next_notification in done:
                translation = tracker.translate(next_notification.result())
                for event in translation.events:
                    yield event
                if translation.completed is not None and translation.completed == active:
                    active = await _deliver(thread, None, queued)
                next_notification = asyncio.ensure_future(thread.notification())
    finally:
        next_notification.cancel()
        next_prompt.cancel()


async def _deliver(
    thread: CodexThread, active: TurnId | None, queued: deque[Prompt]
) -> TurnId | None:
    """Hand `queued` to the running turn, or to a new one if none runs; return the turn.

    A rejected steer leaves the prompt and everything after it queued: the turn may still
    be running, so only its `turn/completed` makes starting another one safe.
    """
    if active is None:
        if not queued:
            return None
        active = await thread.start_turn(queued.popleft())
    while queued:
        try:
            await thread.steer(active, queued[0])
        except RpcError:
            return active
        queued.popleft()
    return active


async def probe_auth(settings: CodexSettings, launch: Launcher = app_server) -> CodexAuth:
    """Report how Codex is authenticated, logging in with the configured API key if it may.

    Unlike a session, the probe logs in again over an API-key login to pick up a rotated key.
    """
    async with launch(_refuse) as connection:
        await handshake(connection)
        auth = auth_state(await _call(connection, Call.ACCOUNT_READ, {}))
        return await ensure_login(connection, settings, auth)


async def ensure_login(
    connection: Connection, settings: CodexSettings, auth: CodexAuth
) -> CodexAuth:
    """Log in with the configured API key if `auth` allows it; return the resulting state.

    The key replaces only a missing or API-key login: logging in rewrites `auth.json`, and
    a ChatGPT login must survive a key left in the environment.
    """
    if settings.api_key is None or not _key_may_replace(auth):
        return auth
    await _call(connection, Call.LOGIN, api_key_login_params(settings.api_key))
    return auth_state(await _call(connection, Call.ACCOUNT_READ, {}))


def _key_may_replace(auth: CodexAuth) -> bool:
    match auth:
        case CodexAuth.MISSING | CodexAuth.API_KEY:
            return True
        case CodexAuth.CHATGPT | CodexAuth.OTHER | CodexAuth.NOT_REQUIRED:
            return False
        case _:
            assert_never(auth)


async def _refuse(method: str, _params: JsonObject) -> JsonObject:
    raise UnsupportedRequestError(method)
