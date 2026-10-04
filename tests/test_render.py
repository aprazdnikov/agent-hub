from decimal import Decimal

import pytest

from agent_hub.render import format_abandoned, format_finished, split_message, truncate


def test_short_text_is_one_chunk() -> None:
    assert split_message("  hello  ") == ["hello"]


def test_empty_text_has_no_chunks() -> None:
    assert split_message("   ") == []


def test_long_text_splits_on_line_boundary() -> None:
    text = "aaaa\nbbbb\ncccc"
    assert split_message(text, limit=9) == ["aaaa\nbbbb", "cccc"]


def test_line_longer_than_limit_is_hard_split() -> None:
    assert split_message("abcdefgh", limit=3) == ["abc", "def", "gh"]


def test_every_chunk_respects_limit() -> None:
    text = "\n".join("x" * n for n in range(1, 60))
    chunks = split_message(text, limit=50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        split_message("x", limit=0)


def test_truncate_marks_cut() -> None:
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("abc", 4) == "abc"


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (None, "✅ Готово · ходов: 3"),
        (Decimal("0.1234"), "✅ Готово · ходов: 3 · $0.12"),
    ],
)
def test_format_finished(cost: Decimal | None, expected: str) -> None:
    assert format_finished(3, cost) == expected


def test_format_finished_mentions_running_background_tasks() -> None:
    assert format_finished(3, None, background=2) == (
        "✅ Готово · ходов: 3 · ⏳ в фоне задач: 2, пришлю результат"
    )


def test_format_abandoned_lists_tasks() -> None:
    assert format_abandoned(("sleep 600", "npm test"), timeout_seconds=1800) == (
        "⌛ Фоновые задачи не завершились за 30 мин, сессия закрыта:\n• sleep 600\n• npm test"
    )


def test_format_finished_without_turns_shows_tokens() -> None:
    assert format_finished(None, None, tokens=12345) == "✅ Готово · токенов в сессии: 12 345"


def test_format_finished_with_nothing_measured() -> None:
    assert format_finished(None, None) == "✅ Готово"
