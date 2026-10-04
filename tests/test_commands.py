from pathlib import Path

import pytest

from agent_hub.commands import (
    AlreadySelected,
    NewSessionArgs,
    ShowBackend,
    ShowSession,
    SwitchBackend,
    UnknownBackend,
    decide_backend,
    join_path_args,
    parse_backend_args,
    parse_new_args,
)
from agent_hub.domain import BackendKind, SessionId, TopicSession


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], NewSessionArgs(BackendKind.CLAUDE, None)),
        (["claude"], NewSessionArgs(BackendKind.CLAUDE, None)),
        (["Claude", "shop/backend"], NewSessionArgs(BackendKind.CLAUDE, "shop/backend")),
        (["shop/backend"], NewSessionArgs(BackendKind.CLAUDE, "shop/backend")),
        (["my", "dir"], NewSessionArgs(BackendKind.CLAUDE, "my dir")),
    ],
)
def test_parse_new_args(args: list[str], expected: NewSessionArgs) -> None:
    assert parse_new_args(args, BackendKind.CLAUDE) == expected


def test_join_path_args() -> None:
    assert join_path_args([]) is None
    assert join_path_args(["a", "b"]) == "a b"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ShowBackend()),
        (["codex"], BackendKind.CODEX),
        (["Claude"], BackendKind.CLAUDE),
        (["gemini"], UnknownBackend("gemini")),
        (["codex", "extra"], UnknownBackend("codex extra")),
    ],
)
def test_parse_backend_args(args: list[str], expected: object) -> None:
    assert parse_backend_args(args) == expected


def test_new_with_codex_and_a_codex_default() -> None:
    assert parse_new_args(["codex", "shop"], BackendKind.CLAUDE) == NewSessionArgs(
        BackendKind.CODEX, "shop"
    )
    assert parse_new_args(["shop"], BackendKind.CODEX) == NewSessionArgs(BackendKind.CODEX, "shop")


@pytest.fixture
def current(tmp_path: Path) -> TopicSession:
    return TopicSession(BackendKind.CLAUDE, tmp_path, SessionId("s-1"))


def test_decide_backend_without_args_shows_the_session(current: TopicSession) -> None:
    assert decide_backend([], current) == ShowSession(current)


def test_decide_backend_unknown_name(current: TopicSession) -> None:
    assert decide_backend(["gemini"], current) == UnknownBackend("gemini")


def test_decide_backend_same_backend_is_already_selected(current: TopicSession) -> None:
    assert decide_backend(["Claude"], current) == AlreadySelected(current)


def test_decide_backend_other_backend_switches_and_drops_the_session_id(
    current: TopicSession, tmp_path: Path
) -> None:
    assert decide_backend(["codex"], current) == SwitchBackend(
        TopicSession(BackendKind.CODEX, tmp_path, None)
    )
