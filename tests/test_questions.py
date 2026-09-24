import pytest

from agent_hub.domain import Denied, Question, QuestionOption, TopicKey
from agent_hub.questions import (
    Accepted,
    Decline,
    NothingSelected,
    Pick,
    QuestionAction,
    QuestionId,
    QuestionPress,
    QuestionRegistry,
    SelectionChanged,
    Stale,
    Submit,
    callback_data,
    keyboard,
    parse_callback_data,
    question_html,
)

KEY = TopicKey(-100, 7)
OTHER_KEY = TopicKey(-100, 8)
OPTIONS = (QuestionOption("Кратко", "Только суть"), QuestionOption("Подробно", ""))
SINGLE = Question("Какой формат?", "Формат", OPTIONS, multi_select=False)
MULTI = Question("Какие разделы?", "", OPTIONS, multi_select=True)


@pytest.mark.parametrize("action", [Pick(0), Pick(11), Submit(), Decline()])
def test_callback_data_round_trips(action: QuestionAction) -> None:
    question_id = QuestionId("0123456789abcdef")
    data = callback_data(question_id, action)
    assert len(data.encode()) <= 64  # Telegram callback_data limit
    assert parse_callback_data(data) == QuestionPress(question_id, action)


@pytest.mark.parametrize(
    "data", ["", "qa", "qa:x:", "qa:x:maybe", "qa:x:-1", "ap:x:0", "qa:x:0:1", "qa::0", "qa:x:²"]
)
def test_malformed_callback_data_is_rejected(data: str) -> None:
    assert parse_callback_data(data) is None


async def test_single_choice_pick_answers_with_label() -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, SINGLE)

    assert registry.press(QuestionPress(question_id, Pick(1))) == Accepted()
    assert await future == "Подробно"
    assert registry.press(QuestionPress(question_id, Pick(0))) == Stale()


async def test_multi_choice_toggles_then_submits_in_option_order() -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, MULTI)

    assert registry.press(QuestionPress(question_id, Pick(1))) == SelectionChanged(
        MULTI, frozenset({1})
    )
    assert registry.press(QuestionPress(question_id, Pick(0))) == SelectionChanged(
        MULTI, frozenset({0, 1})
    )
    assert registry.press(QuestionPress(question_id, Pick(1))) == SelectionChanged(
        MULTI, frozenset({0})
    )
    registry.press(QuestionPress(question_id, Pick(1)))
    assert not future.done()

    assert registry.press(QuestionPress(question_id, Submit())) == Accepted()
    assert await future == "Кратко, Подробно"


async def test_multi_choice_submit_requires_selection() -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, MULTI)

    assert registry.press(QuestionPress(question_id, Submit())) == NothingSelected()
    assert not future.done()


@pytest.mark.parametrize("action", [Pick(2), Submit()])
async def test_invalid_press_for_single_choice_is_stale(action: QuestionAction) -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, SINGLE)

    assert registry.press(QuestionPress(question_id, action)) == Stale()
    assert not future.done()


async def test_decline_denies() -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, SINGLE)

    assert registry.press(QuestionPress(question_id, Decline())) == Accepted()
    outcome = await future
    assert isinstance(outcome, Denied)
    assert outcome.reason


async def test_text_reply_answers_pending_question_of_its_topic_only() -> None:
    registry = QuestionRegistry()
    _, future = registry.open(KEY, SINGLE)

    assert registry.reply(OTHER_KEY, "чужой") is False
    assert registry.reply(KEY, "свой вариант") is True
    assert await future == "свой вариант"
    assert registry.reply(KEY, "ещё") is False


async def test_closed_question_is_stale() -> None:
    registry = QuestionRegistry()
    question_id, future = registry.open(KEY, SINGLE)
    registry.close(question_id)

    assert registry.press(QuestionPress(question_id, Pick(0))) == Stale()
    assert registry.reply(KEY, "x") is False
    assert not future.done()


def test_single_choice_keyboard_has_options_and_decline() -> None:
    question_id = QuestionId("id")
    assert keyboard(question_id, SINGLE, frozenset()) == [
        [("Кратко", callback_data(question_id, Pick(0)))],
        [("Подробно", callback_data(question_id, Pick(1)))],
        [("❌ Не отвечать", callback_data(question_id, Decline()))],
    ]


def test_multi_choice_keyboard_marks_selection_and_submits() -> None:
    question_id = QuestionId("id")
    assert keyboard(question_id, MULTI, frozenset({1})) == [
        [("☐ Кратко", callback_data(question_id, Pick(0)))],
        [("☑ Подробно", callback_data(question_id, Pick(1)))],
        [
            ("✅ Готово", callback_data(question_id, Submit())),
            ("❌ Не отвечать", callback_data(question_id, Decline())),
        ],
    ]


def test_question_html_escapes_and_lists_descriptions() -> None:
    question = Question("a < b?", "H&M", OPTIONS, multi_select=False)
    assert question_html(question) == (
        "❓ <b>H&amp;M</b>\n"
        "a &lt; b?\n\n"
        "• <b>Кратко</b> — Только суть\n"
        "• <b>Подробно</b>\n\n"
        "<i>Выберите вариант или напишите свой ответ сообщением.</i>"
    )
