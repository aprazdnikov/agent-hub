from pathlib import Path

import pytest

from agent_hub.attachments import (
    UPLOADS_DIR,
    AttachmentError,
    outgoing_path,
    prepare_upload,
    prompt_text,
    safe_filename,
    upload_path,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("Отчёт за май.xlsx", "Отчёт_за_май.xlsx"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\win.ini", "win.ini"),
        (".env", "env"),
        ("...", "file"),
        ("", "file"),
        (None, "file"),
        ("a:b*c?.txt", "a_b_c_.txt"),
    ],
)
def test_safe_filename(raw: str | None, expected: str) -> None:
    assert safe_filename(raw, "file") == expected


def test_safe_filename_keeps_extension_when_truncating() -> None:
    name = safe_filename("x" * 300 + ".tar.gz", "file")
    assert len(name) <= 100
    assert name.endswith(".gz")


def test_upload_path_stays_in_uploads_dir() -> None:
    cwd = Path("/work/project")
    path = upload_path(cwd, 42, "../../../x.txt")
    assert path == cwd / UPLOADS_DIR / "42-x.txt"


def test_prompt_text_lists_files_after_text() -> None:
    files = [Path("/w/.agent-hub/uploads/1-a.pdf"), Path("/w/.agent-hub/uploads/2-b.csv")]
    assert prompt_text("сравни", files) == (
        "сравни\n\nПриложенные файлы:\n"
        "- /w/.agent-hub/uploads/1-a.pdf\n"
        "- /w/.agent-hub/uploads/2-b.csv"
    )


def test_prompt_text_without_files_is_unchanged() -> None:
    assert prompt_text("привет", []) == "привет"


def test_prompt_text_with_only_files() -> None:
    assert prompt_text("  ", [Path("/w/a")]) == "Приложенные файлы:\n- /w/a"


def test_prepare_upload_creates_ignored_directory(tmp_path: Path) -> None:
    prepare_upload(tmp_path, upload_path(tmp_path, 1, "a"))

    assert (tmp_path / UPLOADS_DIR).is_dir()
    assert (tmp_path / ".agent-hub" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_prepare_upload_rejects_symlink_escape(tmp_path: Path) -> None:
    cwd, outside = tmp_path / "project", tmp_path / "outside"
    cwd.mkdir()
    outside.mkdir()
    (cwd / ".agent-hub").symlink_to(outside)

    with pytest.raises(AttachmentError):
        prepare_upload(cwd, upload_path(cwd, 1, "a"))


def test_prepare_upload_refuses_existing_target(tmp_path: Path) -> None:
    target = upload_path(tmp_path, 5, "a.txt")
    prepare_upload(tmp_path, target)
    (tmp_path / "secret").write_text("x", encoding="utf-8")
    target.symlink_to(tmp_path / "secret")

    with pytest.raises(AttachmentError):
        prepare_upload(tmp_path, target)


def test_outgoing_path_resolves_relative_to_cwd(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "r.pdf").write_bytes(b"%PDF")

    assert outgoing_path(tmp_path, "out/r.pdf", 10) == (tmp_path / "out" / "r.pdf").resolve()
    assert outgoing_path(tmp_path, str(tmp_path / "out" / "r.pdf"), 10).name == "r.pdf"


@pytest.mark.parametrize("raw", ["missing.pdf", "out", "../secret", "/etc/hostname", "link"])
def test_outgoing_path_rejects_unsendable(tmp_path: Path, raw: str) -> None:
    cwd = tmp_path / "project"
    (cwd / "out").mkdir(parents=True)
    (tmp_path / "secret").write_text("x", encoding="utf-8")
    (cwd / "link").symlink_to(tmp_path / "secret")

    with pytest.raises(AttachmentError):
        outgoing_path(cwd, raw, 10)


def test_outgoing_path_rejects_oversized(tmp_path: Path) -> None:
    (tmp_path / "big.bin").write_bytes(b"x" * 11)

    with pytest.raises(AttachmentError):
        outgoing_path(tmp_path, "big.bin", 10)
