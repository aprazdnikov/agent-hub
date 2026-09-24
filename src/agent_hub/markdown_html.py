"""Agent Markdown → the HTML subset Telegram accepts (parse_mode=HTML).

Telegram supports only inline tags plus <pre>/<blockquote>, so block structure is expressed
with text: headings become bold, list items get bullet markers, tables become <pre> grids.
"""

import html
import re
from collections.abc import Iterable, Sequence

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from agent_hub.render import TELEGRAM_TEXT_LIMIT, split_message

# js-default: tables and strikethrough on, raw HTML off (rendered as escaped text).
_PARSER = MarkdownIt("js-default")
_LINK_SCHEMES = ("http://", "https://", "mailto:", "tg://")
_LANGUAGE = re.compile(r"[\w#+.-]+")
_TAG = re.compile(r"<[^>]+>")
_INDENT = "  "  # Telegram trims plain leading spaces
_BLOCK_SEPARATOR = "\n\n"
_RULE = "──────────"


def markdown_to_html_chunks(markdown: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Render `markdown` as Telegram HTML messages, each at most `limit` characters."""
    root = SyntaxTreeNode(_PARSER.parse(markdown))
    blocks = [piece for node in root.children for piece in _top_block(node, limit)]
    return _pack(blocks, limit)


def html_to_plain(text: str) -> str:
    """Fallback when Telegram rejects the markup: drop tags, decode entities."""
    return html.unescape(_TAG.sub("", text))


def _top_block(node: SyntaxTreeNode, limit: int) -> list[str]:
    rendered = _block(node, depth=0)
    if len(rendered) <= limit:
        return [rendered] if rendered else []
    if node.type in {"fence", "code_block"}:
        return _split_code(node.content, node.info, limit)
    return _split_escaped(_plain(node), limit)


def _pack(blocks: Iterable[str], limit: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}{_BLOCK_SEPARATOR}{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = block
    if current:
        chunks.append(current)
    return chunks


def _block(node: SyntaxTreeNode, depth: int) -> str:
    match node.type:
        case "paragraph":
            return _inline_content(node)
        case "heading":
            return f"<b>{_inline_content(node)}</b>"
        case "fence" | "code_block":
            return _pre(node.content, node.info)
        case "blockquote":
            return f"<blockquote>{_blocks(node.children, depth, '\n')}</blockquote>"
        case "bullet_list" | "ordered_list":
            return _list(node, depth)
        case "table":
            return _table(node)
        case "hr":
            return _RULE
        case _:
            return html.escape(_plain(node))


def _blocks(nodes: Sequence[SyntaxTreeNode], depth: int, separator: str) -> str:
    return separator.join(filter(None, (_block(child, depth) for child in nodes)))


def _list(node: SyntaxTreeNode, depth: int) -> str:
    start = _list_start(node)
    lines: list[str] = []
    for offset, item in enumerate(node.children):
        marker = "•" if start is None else f"{start + offset}."
        # Nested lists render their own deeper indentation.
        body = _blocks(item.children, depth + 1, "\n")
        lines.append(f"{_INDENT * depth}{marker} {body}")
    return "\n".join(lines)


def _list_start(node: SyntaxTreeNode) -> int | None:
    if node.type != "ordered_list":
        return None
    start = node.attrs.get("start", 1)
    return start if isinstance(start, int) else int(str(start))


def _table(node: SyntaxTreeNode) -> str:
    sections = node.children
    rows = [
        [_plain(cell).strip() for cell in row.children]
        for part in sections
        for row in part.children
    ]
    if not rows:
        return ""
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]
    widths = [max(len(row[column]) for row in rows) for column in range(columns)]
    lines = [
        " │ ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    header_rows = len(sections[0].children) if sections and sections[0].type == "thead" else 0
    if header_rows:
        lines.insert(header_rows, "─┼─".join("─" * width for width in widths))
    return _pre("\n".join(line.rstrip() for line in lines), "")


def _pre(code: str, info: str) -> str:
    body = html.escape(code.rstrip("\n"))
    language = info.split(maxsplit=1)[0] if info.strip() else ""
    if _LANGUAGE.fullmatch(language):
        return f'<pre><code class="language-{language}">{body}</code></pre>'
    return f"<pre>{body}</pre>"


def _split_code(code: str, info: str, limit: int) -> list[str]:
    overhead = len(_pre("", info))
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in code.rstrip("\n").split("\n"):
        cost = len(html.escape(line)) + 1
        if current and size + cost > limit - overhead:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += cost
    if current:
        pieces.append("\n".join(current))
    result: list[str] = []
    for piece in pieces:
        rendered = _pre(piece, info)
        result.extend([rendered] if len(rendered) <= limit else _split_escaped(piece, limit))
    return result


def _split_escaped(text: str, limit: int) -> list[str]:
    """Split plain text so that each escaped chunk fits; never cuts an entity in half."""
    result: list[str] = []
    pending = split_message(text, limit)
    while pending:
        chunk = pending.pop(0)
        escaped = html.escape(chunk)
        if len(escaped) <= limit:
            result.append(escaped)
        else:
            pending[:0] = split_message(chunk, max(1, len(chunk) // 2))
    return result


def _inline_content(node: SyntaxTreeNode) -> str:
    return "".join(_inline(child) for child in node.children)


def _inline(node: SyntaxTreeNode) -> str:
    match node.type:
        case "inline":
            return _inline_content(node)
        case "text":
            return html.escape(node.content)
        case "softbreak" | "hardbreak":
            return "\n"
        case "code_inline":
            return f"<code>{html.escape(node.content)}</code>"
        case "strong":
            return f"<b>{_inline_content(node)}</b>"
        case "em":
            return f"<i>{_inline_content(node)}</i>"
        case "s":
            return f"<s>{_inline_content(node)}</s>"
        case "link":
            return _link(node)
        case "image":
            return html.escape(f"{_plain(node)} ({node.attrs.get('src', '')})")
        case _:
            return html.escape(_plain(node))


def _link(node: SyntaxTreeNode) -> str:
    label = _inline_content(node)
    href = str(node.attrs.get("href", ""))
    if href.startswith(_LINK_SCHEMES):
        return f'<a href="{html.escape(href, quote=True)}">{label}</a>'
    # Relative or exotic links make Telegram reject the whole message.
    return f"{label} ({html.escape(href)})" if href else label


def _plain(node: SyntaxTreeNode) -> str:
    match node.type:
        case "text" | "code_inline" | "html_inline" | "html_block" | "fence" | "code_block":
            return node.content
        case "softbreak" | "hardbreak":
            return " "
        case _:
            return "".join(_plain(child) for child in node.children)
