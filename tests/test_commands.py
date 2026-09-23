import pytest

from agent_hub.commands import NewSessionArgs, join_path_args, parse_new_args
from agent_hub.domain import BackendKind


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
    assert parse_new_args(args) == expected


def test_join_path_args() -> None:
    assert join_path_args([]) is None
    assert join_path_args(["a", "b"]) == "a b"
