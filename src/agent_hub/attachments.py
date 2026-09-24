"""Where user uploads land in the session directory and how the agent is told about them."""

import re
from collections.abc import Sequence
from pathlib import Path

# Relative to the session cwd, so the agent can read uploads without leaving its project.
UPLOADS_DIR = Path(".agent-hub/uploads")
# Bot API getFile refuses larger files.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_FILENAME_LENGTH = 100
_UNSAFE = re.compile(r"[^\w.-]")
# Keeps uploads out of the project's git status without touching its own .gitignore.
_GITIGNORE = "*\n"


class AttachmentError(Exception):
    """An upload could not be fetched or stored; the message is shown to the user."""


def safe_filename(raw: str | None, fallback: str) -> str:
    """A single path component that cannot traverse, hide, or overflow."""
    base = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE.sub("_", base).lstrip(".")
    if not name.strip("._"):
        return fallback
    if len(name) <= MAX_FILENAME_LENGTH:
        return name
    stem, dot, suffix = name.rpartition(".")
    if not dot or len(suffix) >= MAX_FILENAME_LENGTH // 2:
        return name[:MAX_FILENAME_LENGTH]
    return f"{stem[: MAX_FILENAME_LENGTH - len(suffix) - 1]}.{suffix}"


def upload_path(cwd: Path, message_id: int, filename: str | None) -> Path:
    # Message ids are unique per chat, so uploads from different messages never collide.
    return cwd / UPLOADS_DIR / f"{message_id}-{safe_filename(filename, 'file')}"


def prompt_text(text: str, files: Sequence[Path]) -> str:
    if not files:
        return text
    listing = "\n".join(["Приложенные файлы:", *(f"- {path}" for path in files)])
    return f"{text.strip()}\n\n{listing}" if text.strip() else listing


def prepare_upload(cwd: Path, target: Path) -> None:
    """Make `target` writable without letting a symlink redirect the write outside `cwd`."""
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(cwd.resolve()):
        raise AttachmentError(f"{directory} ведёт за пределы {cwd}")
    # Names are unique per message, so an existing entry was planted, not uploaded.
    if target.is_symlink() or target.exists():
        raise AttachmentError(f"{target} уже существует")
    ignore = cwd / UPLOADS_DIR.parts[0] / ".gitignore"
    if not ignore.exists():
        ignore.write_text(_GITIGNORE, encoding="utf-8")
