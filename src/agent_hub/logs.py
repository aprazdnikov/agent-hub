"""Logging setup: stable message text plus `key=value` structured fields."""

import logging
from typing import override

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime"}
)


class KeyValueFormatter(logging.Formatter):
    @override
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        if not fields:
            return base
        return base + " " + " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler])
    # httpx logs every request URL at INFO, and Bot API URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
