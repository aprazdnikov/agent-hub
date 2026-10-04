"""Pure parsing and decisions for bot commands."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never, final

from agent_hub.domain import BackendKind, TopicSession


@final
@dataclass(frozen=True, slots=True)
class NewSessionArgs:
    backend: BackendKind
    cwd: str | None


def parse_new_args(args: Sequence[str], default: BackendKind) -> NewSessionArgs:
    """`/new [backend] [path]`; a first word that is not a backend name starts the path."""
    if not args:
        return NewSessionArgs(default, None)
    try:
        backend = BackendKind(args[0].lower())
    except ValueError:
        return NewSessionArgs(default, " ".join(args))
    rest = " ".join(args[1:])
    return NewSessionArgs(backend, rest or None)


def join_path_args(args: Sequence[str]) -> str | None:
    joined = " ".join(args).strip()
    return joined or None


@final
@dataclass(frozen=True, slots=True)
class ShowBackend:
    pass


@final
@dataclass(frozen=True, slots=True)
class UnknownBackend:
    name: str


BackendArgs = BackendKind | ShowBackend | UnknownBackend


def parse_backend_args(args: Sequence[str]) -> BackendArgs:
    """`/backend [name]`: no name shows the current backend."""
    match args:
        case []:
            return ShowBackend()
        case [name]:
            try:
                return BackendKind(name.lower())
            except ValueError:
                return UnknownBackend(name)
        case _:
            return UnknownBackend(" ".join(args))


@final
@dataclass(frozen=True, slots=True)
class ShowSession:
    session: TopicSession


@final
@dataclass(frozen=True, slots=True)
class AlreadySelected:
    session: TopicSession


@final
@dataclass(frozen=True, slots=True)
class SwitchBackend:
    session: TopicSession


BackendDecision = ShowSession | UnknownBackend | AlreadySelected | SwitchBackend


def decide_backend(args: Sequence[str], current: TopicSession) -> BackendDecision:
    parsed = parse_backend_args(args)
    match parsed:
        case ShowBackend():
            return ShowSession(current)
        case UnknownBackend():
            return parsed
        case BackendKind() if parsed is current.backend:
            return AlreadySelected(current)
        case BackendKind():
            # One agent cannot continue another's session.
            return SwitchBackend(TopicSession(parsed, current.cwd, None))
        case _:
            assert_never(parsed)
