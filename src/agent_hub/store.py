"""Durable topic → session mapping in a JSON file."""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_hub.domain import BackendKind, SessionId, TopicKey, TopicSession

FORMAT_VERSION = 1


class CorruptStateError(Exception):
    pass


class TopicStore:
    """In-memory mapping persisted after every change.

    Callers run on a single asyncio loop, so no locking is needed.
    """

    def __init__(self, path: Path, topics: Mapping[TopicKey, TopicSession]) -> None:
        self._path = path
        self._topics = dict(topics)

    @classmethod
    def open(cls, path: Path) -> "TopicStore":
        if not path.exists():
            return cls(path, {})
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorruptStateError(f"cannot read {path}") from error
        return cls(path, parse_state(raw))

    def get(self, key: TopicKey) -> TopicSession | None:
        return self._topics.get(key)

    def put(self, key: TopicKey, session: TopicSession) -> None:
        self._topics[key] = session
        self._flush()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(dump_state(self._topics), ensure_ascii=False, indent=2)
        # Write-then-rename so a crash never leaves a truncated state file.
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".topics-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp).replace(self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def dump_state(topics: Mapping[TopicKey, TopicSession]) -> dict[str, Any]:
    return {
        "version": FORMAT_VERSION,
        "topics": [
            {
                "chat_id": key.chat_id,
                "thread_id": key.thread_id,
                "backend": session.backend.value,
                "cwd": str(session.cwd),
                "session_id": session.session_id,
            }
            for key, session in topics.items()
        ],
    }


def parse_state(raw: object) -> dict[TopicKey, TopicSession]:
    if not isinstance(raw, dict) or raw.get("version") != FORMAT_VERSION:
        raise CorruptStateError(f"expected state format version {FORMAT_VERSION}")
    entries = raw.get("topics")
    if not isinstance(entries, list):
        raise CorruptStateError("'topics' must be a list")
    return dict(_parse_entry(entry) for entry in entries)


def _parse_entry(entry: object) -> tuple[TopicKey, TopicSession]:
    if not isinstance(entry, dict):
        raise CorruptStateError(f"topic entry must be an object: {entry!r}")
    chat_id, thread_id = entry.get("chat_id"), entry.get("thread_id")
    backend, cwd, session_id = entry.get("backend"), entry.get("cwd"), entry.get("session_id")
    if not isinstance(chat_id, int) or not isinstance(thread_id, int):
        raise CorruptStateError(f"chat_id and thread_id must be integers: {entry!r}")
    if not isinstance(backend, str) or not isinstance(cwd, str):
        raise CorruptStateError(f"backend and cwd must be strings: {entry!r}")
    if not (session_id is None or isinstance(session_id, str)):
        raise CorruptStateError(f"session_id must be a string or null: {entry!r}")
    try:
        session = TopicSession(
            backend=BackendKind(backend),
            cwd=Path(cwd),
            session_id=None if session_id is None else SessionId(session_id),
        )
    except ValueError as error:
        raise CorruptStateError(f"invalid topic entry: {entry!r}") from error
    return TopicKey(chat_id, thread_id), session
