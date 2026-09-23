import json
from pathlib import Path

import pytest

from agent_hub.domain import BackendKind, SessionId, TopicKey, TopicSession
from agent_hub.store import CorruptStateError, TopicStore


def test_missing_file_is_empty_store(tmp_path: Path) -> None:
    store = TopicStore.open(tmp_path / "state" / "topics.json")
    assert store.get(TopicKey(1, 2)) is None


def test_put_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state" / "topics.json"
    key = TopicKey(-100123, 42)
    session = TopicSession(BackendKind.CLAUDE, tmp_path, SessionId("abc"))

    TopicStore.open(path).put(key, session)

    assert TopicStore.open(path).get(key) == session
    assert not list(path.parent.glob(".topics-*")), "temporary files must be cleaned up"


def test_session_without_id_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "topics.json"
    key = TopicKey(1, 2)
    session = TopicSession(BackendKind.CLAUDE, tmp_path, None)

    TopicStore.open(path).put(key, session)

    assert TopicStore.open(path).get(key) == session


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 999, "topics": []}),
        json.dumps({"version": 1, "topics": {}}),
        json.dumps({"version": 1, "topics": [{"chat_id": "1"}]}),
        json.dumps(
            {
                "version": 1,
                "topics": [
                    {"chat_id": 1, "thread_id": 2, "backend": "gpt", "cwd": "/", "session_id": None}
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "topics": [
                    {
                        "chat_id": 1,
                        "thread_id": 2,
                        "backend": "claude",
                        "cwd": "rel",
                        "session_id": None,
                    }
                ],
            }
        ),
    ],
)
def test_corrupt_state_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "topics.json"
    path.write_text(content)
    with pytest.raises(CorruptStateError):
        TopicStore.open(path)
