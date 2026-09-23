"""Pure parsing of bot command arguments."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

from agent_hub.domain import BackendKind

DEFAULT_BACKEND = BackendKind.CLAUDE


@final
@dataclass(frozen=True, slots=True)
class NewSessionArgs:
    backend: BackendKind
    cwd: str | None


def parse_new_args(args: Sequence[str]) -> NewSessionArgs:
    """`/new [backend] [path]`; a first word that is not a backend name starts the path."""
    if not args:
        return NewSessionArgs(DEFAULT_BACKEND, None)
    try:
        backend = BackendKind(args[0].lower())
    except ValueError:
        return NewSessionArgs(DEFAULT_BACKEND, " ".join(args))
    rest = " ".join(args[1:])
    return NewSessionArgs(backend, rest or None)


def join_path_args(args: Sequence[str]) -> str | None:
    joined = " ".join(args).strip()
    return joined or None
