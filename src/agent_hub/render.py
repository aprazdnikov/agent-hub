"""Pure formatting of agent output for Telegram."""

from collections.abc import Sequence
from decimal import Decimal

TELEGRAM_TEXT_LIMIT = 4096


def split_message(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split `text` into chunks of at most `limit` chars, preferring line boundaries."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    chunks: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        chunks.append(rest)
    return chunks


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_finished(
    turns: int | None, cost_usd: Decimal | None, background: int = 0, tokens: int | None = None
) -> str:
    parts = ["✅ Готово"]
    if turns is not None:
        parts.append(f"ходов: {turns}")
    if cost_usd is not None:
        parts.append(f"${cost_usd.quantize(Decimal('0.01'))}")
    if tokens is not None:
        parts.append(f"токенов в сессии: {tokens:_}".replace("_", " "))
    if background:
        parts.append(f"⏳ в фоне задач: {background}, пришлю результат")
    return " · ".join(parts)


def format_abandoned(tasks: Sequence[str], timeout_seconds: int) -> str:
    listing = "\n".join(f"• {task}" for task in tasks)
    return (
        f"⌛ Фоновые задачи не завершились за {timeout_seconds // 60} мин, сессия закрыта:\n"
        f"{listing}"
    )
