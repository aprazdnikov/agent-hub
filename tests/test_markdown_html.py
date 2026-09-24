import re

import pytest

from agent_hub.markdown_html import html_to_plain, markdown_to_html_chunks

_ALLOWED_TAG = re.compile(r"</?(b|i|s|code|pre|blockquote|a)(\s[^>]*)?>")


def render(markdown: str) -> str:
    chunks = markdown_to_html_chunks(markdown)
    assert len(chunks) == 1
    return chunks[0]


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("**bold** and *italic*", "<b>bold</b> and <i>italic</i>"),
        ("~~gone~~", "<s>gone</s>"),
        ("run `uv sync`", "run <code>uv sync</code>"),
        ("# Title", "<b>Title</b>"),
        ("a < b & c > d", "a &lt; b &amp; c &gt; d"),
        ("<script>x</script>", "&lt;script&gt;x&lt;/script&gt;"),
        ("---", "──────────"),
    ],
)
def test_inline_and_simple_blocks(markdown: str, expected: str) -> None:
    assert render(markdown) == expected


def test_fenced_code_keeps_language_and_escapes() -> None:
    assert render("```python\nif a < b:\n    pass\n```") == (
        '<pre><code class="language-python">if a &lt; b:\n    pass</code></pre>'
    )


def test_fence_without_language_is_plain_pre() -> None:
    assert render("```\nls\n```") == "<pre>ls</pre>"


def test_lists_get_markers_and_nesting() -> None:
    assert render("- one\n- two\n  - nested\n\n3. three\n4. four") == (
        "• one\n• two\n\u00a0\u00a0• nested\n\n3. three\n4. four"
    )


def test_links_keep_safe_schemes_only() -> None:
    assert render("[docs](https://example.com/?a=1&b=2)") == (
        '<a href="https://example.com/?a=1&amp;b=2">docs</a>'
    )
    assert render("[file](src/app.py)") == "file (src/app.py)"
    # markdown-it refuses dangerous schemes itself and leaves the source as text.
    assert render("[x](javascript:alert(1))") == "[x](javascript:alert(1))"


def test_table_becomes_aligned_pre() -> None:
    assert render("| a | long |\n|---|---|\n| xx | y |") == (
        "<pre>a  │ long\n───┼─────\nxx │ y</pre>"
    )


def test_blockquote() -> None:
    assert render("> quoted **text**") == "<blockquote>quoted <b>text</b></blockquote>"


def test_blocks_are_separated_by_blank_line() -> None:
    assert render("para one\n\npara two") == "para one\n\npara two"


def test_long_text_is_split_into_valid_chunks() -> None:
    markdown = "\n\n".join(f"**{i}** " + "word & " * 40 for i in range(60))
    chunks = markdown_to_html_chunks(markdown, limit=500)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500
        assert chunk.count("<b>") == chunk.count("</b>")


def test_huge_code_block_is_split_into_several_pre() -> None:
    code = "\n".join(f"line {i} <tag>" for i in range(200))
    chunks = markdown_to_html_chunks(f"```\n{code}\n```", limit=400)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 400
        assert chunk.startswith("<pre>")
        assert chunk.endswith("</pre>")
    assert html_to_plain("\n".join(chunks)) == code


def test_oversized_paragraph_falls_back_to_escaped_text() -> None:
    chunks = markdown_to_html_chunks("&" * 300, limit=100)

    assert all(len(chunk) <= 100 for chunk in chunks)
    assert html_to_plain("".join(chunks)).replace("\n", "") == "&" * 300


def test_only_telegram_tags_are_emitted() -> None:
    markdown = (
        "# H\n\n**b** *i* ~~s~~ `c` [l](https://x.y)\n\n> q\n\n- a\n\n"
        "```js\nx\n```\n\n|a|b|\n|-|-|\n|1|2|"
    )
    tags = re.findall(r"</?[^>]+>", render(markdown))
    assert tags
    assert all(_ALLOWED_TAG.fullmatch(tag) for tag in tags)


def test_empty_markdown_has_no_chunks() -> None:
    assert markdown_to_html_chunks("   ") == []


def test_html_to_plain() -> None:
    assert html_to_plain("<b>a &amp; b</b> <code>&lt;x&gt;</code>") == "a & b <x>"
