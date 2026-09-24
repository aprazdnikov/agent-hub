"""Pending agent questions answered with buttons or a free-text message in the same topic."""

import asyncio
import html
import secrets
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NewType, assert_never, final

from agent_hub.domain import Denied, Question, TopicKey
from agent_hub.render import truncate

QuestionId = NewType("QuestionId", str)

CALLBACK_PREFIX = "qa"
_CALLBACK_PARTS = 3  # prefix:id:action
_SUBMIT = "done"
_DECLINE = "no"
DECLINED_BY_USER = "Пользователь отказался отвечать на вопрос"
BUTTON_TEXT_LIMIT = 60
# Keep the whole question inside one Telegram message.
QUESTION_TEXT_LIMIT = 2000
DESCRIPTION_TEXT_LIMIT = 300
# One answer per option label; multi-select answers use the separator Claude Code expects.
ANSWER_SEPARATOR = ", "

Button = tuple[str, str]  # text, callback data
QuestionAnswer = str | Denied


@final
@dataclass(frozen=True, slots=True)
class Pick:
    index: int


@final
@dataclass(frozen=True, slots=True)
class Submit:
    pass


@final
@dataclass(frozen=True, slots=True)
class Decline:
    pass


QuestionAction = Pick | Submit | Decline


@final
@dataclass(frozen=True, slots=True)
class QuestionPress:
    question_id: QuestionId
    action: QuestionAction


@final
@dataclass(frozen=True, slots=True)
class Accepted:
    pass


@final
@dataclass(frozen=True, slots=True)
class SelectionChanged:
    question: Question
    selected: frozenset[int]


@final
@dataclass(frozen=True, slots=True)
class NothingSelected:
    pass


@final
@dataclass(frozen=True, slots=True)
class Stale:
    """The question is gone (answered, timed out, stopped) or the button does not fit it."""


PressResult = Accepted | SelectionChanged | NothingSelected | Stale


@dataclass(slots=True)
class _Pending:
    key: TopicKey
    question: Question
    future: asyncio.Future[QuestionAnswer]
    selected: frozenset[int] = field(default_factory=frozenset)


class QuestionRegistry:
    def __init__(self) -> None:
        self._pending: dict[QuestionId, _Pending] = {}

    def open(
        self, key: TopicKey, question: Question
    ) -> tuple[QuestionId, asyncio.Future[QuestionAnswer]]:
        question_id = QuestionId(secrets.token_hex(8))
        future: asyncio.Future[QuestionAnswer] = asyncio.get_running_loop().create_future()
        self._pending[question_id] = _Pending(key, question, future)
        return question_id, future

    def close(self, question_id: QuestionId) -> None:
        self._pending.pop(question_id, None)

    def press(self, press: QuestionPress) -> PressResult:
        pending = self._pending.get(press.question_id)
        if pending is None or pending.future.done():
            return Stale()
        question = pending.question
        match press.action:
            case Pick(index) if not 0 <= index < len(question.options):
                return Stale()
            case Pick(index) if question.multi_select:
                pending.selected ^= {index}
                return SelectionChanged(question, pending.selected)
            case Pick(index):
                return self._resolve(press.question_id, question.options[index].label)
            case Submit() if not question.multi_select:
                return Stale()
            case Submit() if not pending.selected:
                return NothingSelected()
            case Submit():
                labels = (question.options[index].label for index in sorted(pending.selected))
                return self._resolve(press.question_id, ANSWER_SEPARATOR.join(labels))
            case Decline():
                return self._resolve(press.question_id, Denied(DECLINED_BY_USER))
            case _:
                assert_never(press.action)

    def reply(self, key: TopicKey, text: str) -> bool:
        """Answer the topic's open question with free text; False when none is open."""
        question_id = next(self._open_in(key), None)
        if question_id is None:
            return False
        self._resolve(question_id, text)
        return True

    def _open_in(self, key: TopicKey) -> Iterator[QuestionId]:
        return (
            question_id
            for question_id, pending in self._pending.items()
            if pending.key == key and not pending.future.done()
        )

    def _resolve(self, question_id: QuestionId, answer: QuestionAnswer) -> Accepted:
        self._pending.pop(question_id).future.set_result(answer)
        return Accepted()


def callback_data(question_id: QuestionId, action: QuestionAction) -> str:
    match action:
        case Pick(index):
            code = str(index)
        case Submit():
            code = _SUBMIT
        case Decline():
            code = _DECLINE
        case _:
            assert_never(action)
    return f"{CALLBACK_PREFIX}:{question_id}:{code}"


def parse_callback_data(data: str) -> QuestionPress | None:
    parts = data.split(":")
    if len(parts) != _CALLBACK_PARTS or parts[0] != CALLBACK_PREFIX or not parts[1]:
        return None
    action = _parse_action(parts[2])
    return None if action is None else QuestionPress(QuestionId(parts[1]), action)


def _parse_action(code: str) -> QuestionAction | None:
    if code == _SUBMIT:
        return Submit()
    if code == _DECLINE:
        return Decline()
    return Pick(int(code)) if code.isascii() and code.isdigit() else None


def keyboard(
    question_id: QuestionId, question: Question, selected: frozenset[int]
) -> list[list[Button]]:
    rows = [
        [(_option_text(question, index, selected), callback_data(question_id, Pick(index)))]
        for index in range(len(question.options))
    ]
    decline = ("❌ Не отвечать", callback_data(question_id, Decline()))
    if question.multi_select:
        return [*rows, [("✅ Готово", callback_data(question_id, Submit())), decline]]
    return [*rows, [decline]]


def _option_text(question: Question, index: int, selected: frozenset[int]) -> str:
    label = truncate(question.options[index].label, BUTTON_TEXT_LIMIT)
    if not question.multi_select:
        return label
    return f"{'☑' if index in selected else '☐'} {label}"


def question_html(question: Question) -> str:
    title = f"❓ <b>{html.escape(question.header)}</b>\n" if question.header.strip() else "❓ "
    options = "\n".join(
        f"• <b>{html.escape(option.label)}</b>"
        + (
            f" — {html.escape(truncate(option.description, DESCRIPTION_TEXT_LIMIT))}"
            if option.description.strip()
            else ""
        )
        for option in question.options
    )
    hint = (
        "Отметьте варианты и нажмите «Готово» или напишите свой ответ сообщением."
        if question.multi_select
        else "Выберите вариант или напишите свой ответ сообщением."
    )
    text = html.escape(truncate(question.text, QUESTION_TEXT_LIMIT))
    parts = [f"{title}{text}", options, f"<i>{hint}</i>"]
    return "\n\n".join(part for part in parts if part)
