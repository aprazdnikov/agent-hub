"""Newline-delimited JSON-RPC 2.0 over a child process's stdio, in both directions."""

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, final

log = logging.getLogger(__name__)

JsonObject = dict[str, Any]
RequestId = int | str

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """The peer answered a request with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class TransportClosedError(Exception):
    """The peer closed its output; nothing more will arrive."""


class ProtocolError(Exception):
    """The peer sent a message of an unexpected shape."""


class UnsupportedRequestError(Exception):
    """A server request this client does not implement."""


class MalformedRequestError(Exception):
    """A server request whose params do not have the expected shape."""


@final
@dataclass(frozen=True, slots=True)
class Notification:
    method: str
    params: JsonObject


RequestHandler = Callable[[str, JsonObject], Awaitable[JsonObject]]


class LineReader(Protocol):
    async def readline(self) -> bytes: ...


class LineWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


class Connection(Protocol):
    """What a session needs from a JSON-RPC peer."""

    async def request(self, method: str, params: JsonObject) -> JsonObject: ...

    async def notify(self, method: str) -> None: ...

    async def notification(self) -> Notification: ...


class RpcConnection:
    """Matches responses to requests, queues notifications and answers server requests.

    Each server request runs in its own task, so one awaiting a human does not stall the
    stream. Progress requires `serve`; once it returns, every call raises
    `TransportClosedError`.
    """

    def __init__(self, reader: LineReader, writer: LineWriter, handler: RequestHandler) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._last_id = 0
        self._pending: dict[RequestId, asyncio.Future[JsonObject]] = {}
        self._notifications: asyncio.Queue[Notification | None] = asyncio.Queue()
        self._answering: set[asyncio.Task[None]] = set()
        self._closed = False

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        self._last_id += 1
        request_id = self._last_id
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str) -> None:
        await self._send({"method": method})

    async def notification(self) -> Notification:
        item = await self._notifications.get()
        if item is None:
            # Keep the end marker for any later reader.
            self._notifications.put_nowait(None)
            raise TransportClosedError("app-server closed its output")
        return item

    async def serve(self) -> None:
        try:
            while line := await self._reader.readline():
                self._dispatch(line)
        finally:
            self._close()

    def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            log.warning("unparsable rpc line", extra={"line": line[:200]})
            return
        match message:
            case {"id": int() | str() as request_id, "method": str() as method}:
                task = asyncio.create_task(self._answer(request_id, method, _params(message)))
                self._answering.add(task)
                task.add_done_callback(self._answering.discard)
            case {"method": str() as method}:
                self._notifications.put_nowait(Notification(method, _params(message)))
            case {"id": int() | str() as request_id}:
                self._resolve(request_id, message)
            case _:
                log.warning("unexpected rpc message", extra={"line": line[:200]})

    def _resolve(self, request_id: RequestId, message: JsonObject) -> None:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        match message:
            case {"error": {"code": int() as code, "message": str() as text}}:
                future.set_exception(RpcError(code, text))
            case {"result": dict() as result}:
                future.set_result(result)
            case {"result": None}:
                future.set_result({})
            case _:
                future.set_exception(ProtocolError(f"malformed response: {message!r}"))

    async def _answer(self, request_id: RequestId, method: str, params: JsonObject) -> None:
        try:
            reply: JsonObject = {"id": request_id, "result": await self._handler(method, params)}
        except UnsupportedRequestError:
            reply = _error(request_id, METHOD_NOT_FOUND, f"unsupported: {method}")
        except MalformedRequestError as error:
            reply = _error(request_id, INVALID_PARAMS, str(error))
        except Exception:
            log.exception("server request failed", extra={"method": method})
            reply = _error(request_id, INTERNAL_ERROR, "agent-hub failed to handle the request")
        # The peer is gone; there is nobody left to answer.
        with contextlib.suppress(TransportClosedError):
            await self._send(reply)

    async def _send(self, message: JsonObject) -> None:
        if self._closed:
            raise TransportClosedError("app-server closed its output")
        try:
            self._writer.write(json.dumps(message).encode() + b"\n")
            await self._writer.drain()
        except OSError as error:
            raise TransportClosedError("app-server closed its input") from error

    def _close(self) -> None:
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(TransportClosedError("app-server closed its output"))
        for task in self._answering:
            task.cancel()
        self._notifications.put_nowait(None)


def _params(message: JsonObject) -> JsonObject:
    params = message.get("params")
    return params if isinstance(params, dict) else {}


def _error(request_id: RequestId, code: int, message: str) -> JsonObject:
    return {"id": request_id, "error": {"code": code, "message": message}}
