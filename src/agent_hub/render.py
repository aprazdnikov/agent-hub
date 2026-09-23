"""Pure formatting of agent output for Telegram."""

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


def format_finished(turns: int, cost_usd: Decimal | None) -> str:
    cost = "" if cost_usd is None else f" · ${cost_usd.quantize(Decimal('0.01'))}"
    return f"✅ Готово · ходов: {turns}{cost}"
