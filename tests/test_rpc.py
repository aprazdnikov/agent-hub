import asyncio
import gc
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import override

import pytest

from agent_hub.backends.rpc import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    JsonObject,
    MalformedRequestError,
    Notification,
    RequestHandler,
    RpcConnection,
    RpcError,
    TransportClosedError,
    UnsupportedRequestError,
)


class FakeWriter:
    def __init__(self) -> None:
        self.sent: asyncio.Queue[JsonObject] = asyncio.Queue()

    def write(self, data: bytes) -> None:
        self.sent.put_nowait(json.loads(data))

    async def drain(self) -> None:
        return None


async def _echo(method: str, params: JsonObject) -> JsonObject:
    match method:
        case "echo":
            return params
        case "malformed":
            raise MalformedRequestError("bad params")
        case _:
            raise UnsupportedRequestError(method)


class Peer:
    """The app-server end of a connection under test."""

    def __init__(self, handler: RequestHandler) -> None:
        self.reader = asyncio.StreamReader()
        self.writer = FakeWriter()
        self.connection = RpcConnection(self.reader, self.writer, handler)

    def send(self, message: JsonObject) -> None:
        self.reader.feed_data(json.dumps(message).encode() + b"\n")

    async def received(self) -> JsonObject:
        return await asyncio.wait_for(self.writer.sent.get(), 1)


@asynccontextmanager
async def serving(handler: RequestHandler = _echo) -> AsyncIterator[Peer]:
    peer = Peer(handler)
    serve = asyncio.create_task(peer.connection.serve())
    try:
        yield peer
    finally:
        peer.reader.feed_eof()
        await asyncio.wait_for(serve, 1)


async def test_request_returns_the_matching_result() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("thread/start", {"cwd": "w"}))

        assert await peer.received() == {"id": 1, "method": "thread/start", "params": {"cwd": "w"}}
        peer.send({"id": 1, "result": {"thread": {"id": "t1"}}})

        assert await asyncio.wait_for(call, 1) == {"thread": {"id": "t1"}}


async def test_error_response_raises() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("thread/resume", {}))
        await peer.received()
        peer.send({"id": 1, "error": {"code": -32600, "message": "no rollout"}})

        with pytest.raises(RpcError) as raised:
            await asyncio.wait_for(call, 1)
        assert (raised.value.code, raised.value.message) == (-32600, "no rollout")


async def test_notifications_arrive_in_order() -> None:
    async with serving() as peer:
        peer.reader.feed_data(b"not json\n")
        peer.reader.feed_data(b"\x80\x80\n")
        peer.send({"method": "turn/started", "params": {"turn": {"id": "a"}}})
        peer.send({"method": "turn/completed"})

        assert await peer.connection.notification() == Notification(
            "turn/started", {"turn": {"id": "a"}}
        )
        assert await peer.connection.notification() == Notification("turn/completed", {})


async def test_notify_has_no_id() -> None:
    async with serving() as peer:
        await peer.connection.notify("initialized")
        assert await peer.received() == {"method": "initialized"}


async def test_server_request_is_answered_by_the_handler() -> None:
    async with serving() as peer:
        peer.send({"id": 7, "method": "echo", "params": {"a": 1}})
        assert await peer.received() == {"id": 7, "result": {"a": 1}}


@pytest.mark.parametrize(
    ("method", "code"), [("unknown/method", METHOD_NOT_FOUND), ("malformed", INVALID_PARAMS)]
)
async def test_refused_server_request_gets_an_error(method: str, code: int) -> None:
    async with serving() as peer:
        peer.send({"id": 8, "method": method, "params": {}})
        reply = await peer.received()
        assert reply["id"] == 8
        assert reply["error"]["code"] == code


async def test_pending_server_request_does_not_block_responses() -> None:
    release = asyncio.Event()

    async def slow(method: str, _params: JsonObject) -> JsonObject:
        await release.wait()
        return {"method": method}

    async with serving(slow) as peer:
        peer.send({"id": 1, "method": "approve", "params": {}})
        call = asyncio.create_task(peer.connection.request("turn/steer", {}))
        request = await peer.received()
        peer.send({"id": request["id"], "result": {"turnId": "t"}})

        assert await asyncio.wait_for(call, 1) == {"turnId": "t"}
        release.set()
        assert await peer.received() == {"id": 1, "result": {"method": "approve"}}


async def test_closed_transport_fails_pending_and_later_calls() -> None:
    async with serving() as peer:
        call = asyncio.create_task(peer.connection.request("turn/start", {}))
        await peer.received()
        peer.reader.feed_eof()

        with pytest.raises(TransportClosedError):
            await asyncio.wait_for(call, 1)
        with pytest.raises(TransportClosedError):
            await peer.connection.notification()
        with pytest.raises(TransportClosedError):
            await peer.connection.notification()
        with pytest.raises(TransportClosedError):
            await peer.connection.request("turn/start", {})


class BrokenWriter(FakeWriter):
    @override
    async def drain(self) -> None:
        raise BrokenPipeError


async def test_broken_pipe_is_a_closed_transport() -> None:
    reader = asyncio.StreamReader()
    connection = RpcConnection(reader, BrokenWriter(), _echo)
    serve = asyncio.create_task(connection.serve())
    try:
        reader.feed_data(b'{"id": 3, "method": "echo", "params": {}}\n')
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with pytest.raises(TransportClosedError):
            await connection.request("turn/start", {})
        with pytest.raises(TransportClosedError):
            await connection.notify("initialized")
    finally:
        reader.feed_eof()
        await asyncio.wait_for(serve, 1)
    await asyncio.sleep(0)
    gc.collect()
