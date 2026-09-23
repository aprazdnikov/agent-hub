"""Resolution of user-supplied working directories, confined to the workspace root."""

from pathlib import Path


class InvalidCwdError(Exception):
    pass


def resolve_cwd(root: Path, raw: str | None) -> Path:
    """Resolve `raw` (absolute, `~`-prefixed or relative to `root`) inside `root`.

    Symlinks are resolved before the containment check so a link cannot escape the root.
    """
    if raw is None or not raw.strip():
        return root
    candidate = Path(raw.strip()).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise InvalidCwdError(f"{resolved} is outside workspace root {root}")
    if not resolved.is_dir():
        raise InvalidCwdError(f"{resolved} is not a directory")
    return resolved
