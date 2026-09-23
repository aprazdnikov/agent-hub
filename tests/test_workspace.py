from pathlib import Path

import pytest

from agent_hub.workspace import InvalidCwdError, resolve_cwd


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "project" / "sub").mkdir(parents=True)
    (tmp_path / "file.txt").write_text("x")
    return tmp_path.resolve()


def test_missing_path_means_root(root: Path) -> None:
    assert resolve_cwd(root, None) == root
    assert resolve_cwd(root, "  ") == root


def test_relative_path_is_under_root(root: Path) -> None:
    assert resolve_cwd(root, "project/sub") == root / "project" / "sub"


def test_absolute_path_inside_root_is_accepted(root: Path) -> None:
    assert resolve_cwd(root, str(root / "project")) == root / "project"


@pytest.mark.parametrize("raw", ["..", "../..", "/", "project/../../"])
def test_escape_from_root_is_rejected(root: Path, raw: str) -> None:
    with pytest.raises(InvalidCwdError):
        resolve_cwd(root, raw)


def test_symlink_escape_is_rejected(root: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (root / "link").symlink_to(outside)
    with pytest.raises(InvalidCwdError):
        resolve_cwd(root, "link")


@pytest.mark.parametrize("raw", ["missing", "file.txt"])
def test_non_directory_is_rejected(root: Path, raw: str) -> None:
    with pytest.raises(InvalidCwdError):
        resolve_cwd(root, raw)
