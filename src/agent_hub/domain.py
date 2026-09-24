"""Backend-neutral domain types shared by the bot and agent backends."""

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import NewType, final

SessionId = NewType("SessionId", str)


class BackendKind(StrEnum):
    CLAUDE = "claude"


@final
@dataclass(frozen=True, slots=True)
class TopicKey:
    """One forum topic of one Telegram chat."""

    chat_id: int
    thread_id: int


@final
@dataclass(frozen=True, slots=True)
class TopicSession:
    backend: BackendKind
    cwd: Path
    session_id: SessionId | None

    def __post_init__(self) -> None:
        if not self.cwd.is_absolute():
            raise ValueError(f"cwd must be absolute: {self.cwd}")

    def with_session(self, session_id: SessionId | None) -> "TopicSession":
        return replace(self, session_id=session_id)


@final
@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A tool call the agent wants to make, awaiting a human decision."""

    tool: str
    summary: str


@final
@dataclass(frozen=True, slots=True)
class Allowed:
    pass


@final
@dataclass(frozen=True, slots=True)
class Denied:
    reason: str


Decision = Allowed | Denied


@final
@dataclass(frozen=True, slots=True)
class QuestionOption:
    label: str
    description: str


@final
@dataclass(frozen=True, slots=True)
class Question:
    """A clarifying question the agent asks, answered by an option label or free text."""

    text: str
    header: str
    options: tuple[QuestionOption, ...]
    multi_select: bool

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("question text must not be empty")


@final
@dataclass(frozen=True, slots=True)
class Answered:
    """Answers keyed by question text; multi-select labels are joined with ", "."""

    answers: tuple[tuple[str, str], ...]


QuestionsOutcome = Answered | Denied


@final
@dataclass(frozen=True, slots=True)
class OutgoingFile:
    """A file the agent asks to deliver; `path` is as the agent wrote it, not yet checked."""

    path: str
    caption: str


@final
@dataclass(frozen=True, slots=True)
class Delivered:
    pass


FileDelivery = Delivered | Denied


class ImageMediaType(StrEnum):
    """Image formats the Claude API accepts inline."""

    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"


@final
@dataclass(frozen=True, slots=True)
class Image:
    media_type: ImageMediaType
    data: bytes


@final
@dataclass(frozen=True, slots=True)
class Prompt:
    """One user turn: text plus images sent inline to the model."""

    text: str
    images: tuple[Image, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip() and not self.images:
            raise ValueError("prompt must have text or images")


@final
@dataclass(frozen=True, slots=True)
class SessionStarted:
    session_id: SessionId


@final
@dataclass(frozen=True, slots=True)
class AssistantText:
    text: str


@final
@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    summary: str


@final
@dataclass(frozen=True, slots=True)
class Finished:
    session_id: SessionId
    turns: int
    cost_usd: Decimal | None


@final
@dataclass(frozen=True, slots=True)
class Failed:
    reason: str


AgentEvent = SessionStarted | AssistantText | ToolCall | Finished | Failed
