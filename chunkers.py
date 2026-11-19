"""Chunking strategies for markdown ingestion.

chunk_fixed: fixed-size windows with overlap (verbatim from original main.py).
chunk_recursive: split on nested separators, greedy-merge to max_chars.
chunk_markdown: section-aware chunking returning (content, heading_path).
chunk_markdown_metadata: same, as MarkdownChunk records carrying the span
(char/line offsets) of each chunk in the original text.
"""

import re
from dataclasses import dataclass

# Separators for chunk_recursive, coarsest first.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]

# ATX heading: 1-6 '#' followed by whitespace and a title.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def chunk_fixed(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Step 1: Split markdown text into fixed-size chunks with character overlap to preserve code context."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            next_newline = text.rfind("\n", start, end)
            if next_newline > start + (max_chars // 2):
                end = next_newline + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap

    return chunks


def _recursive_split(text: str, separators: list[str], max_chars: int) -> list[str]:
    """Split text on separators[0] (keeping the separator text so chunks
    reconstruct the source), recursing on pieces still over max_chars.
    With no separators left, hard-truncate the atomic piece."""
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[:max_chars]]

    sep, rest = separators[0], separators[1:]
    pattern = f"({re.escape(sep)})"
    pieces = []
    for part in re.split(pattern, text):
        if not part:
            continue
        if len(part) <= max_chars:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, rest, max_chars))
    return pieces


def chunk_recursive(text: str, max_chars: int = 800) -> list[str]:
    """Split on nested separators, then greedily merge pieces up to max_chars."""
    pieces = _recursive_split(text, _RECURSIVE_SEPARATORS, max_chars)

    chunks = []
    cur = ""
    for piece in pieces:
        if len(cur) + len(piece) <= max_chars:
            cur += piece
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _merge_paragraphs(paras: list[str], max_chars: int) -> list[str]:
    """Greedily merge paragraphs up to max_chars; hard-truncate oversized ones."""
    chunks = []
    cur = None
    for para in paras:
        if len(para) > max_chars:
            para = para[:max_chars]
        if cur is None:
            cur = para
        elif len(cur) + 2 + len(para) <= max_chars:  # 2 = len("\n\n")
            cur = cur + "\n\n" + para
        else:
            chunks.append(cur)
            cur = para
    if cur is not None and cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def _merge_paras(paras: list[tuple[str, int, int, int]], max_chars: int) -> list[tuple[str, int, int, int]]:
    """Greedily merge (text, char_start, char_end, line_start) paragraphs up to
    max_chars; hard-truncate oversized ones. Same rule as _merge_paragraphs,
    carrying the covering span (char_end tracks the last merged paragraph)."""
    chunks = []
    cur = None
    for text, char_start, char_end, line_start in paras:
        if len(text) > max_chars:
            text, char_end = text[:max_chars], char_start + max_chars
        if cur is None:
            cur = (text, char_start, char_end, line_start)
        elif len(cur[0]) + 2 + len(text) <= max_chars:  # 2 = len("\n\n")
            cur = (cur[0] + "\n\n" + text, cur[1], char_end, cur[3])
        else:
            chunks.append(cur)
            cur = (text, char_start, char_end, line_start)
    if cur is not None and cur[0].strip():
        chunks.append(cur)
    return [(c.strip(), cs, ce, ls) for c, cs, ce, ls in chunks if c.strip()]


@dataclass(frozen=True)
class MarkdownChunk:
    """One markdown chunk + its span in the original file.

    char_start/char_end cover the merged paragraphs (including the original
    blank-line separators between them); content is the stripped, normalized
    text, so for multi-paragraph chunks content != text[char_start:char_end].
    line_start is the 1-based line of the chunk's first line."""

    content: str
    heading_path: str | None
    char_start: int
    char_end: int
    line_start: int


def chunk_markdown_metadata(text: str, max_chars: int = 800) -> list[MarkdownChunk]:
    """Offset-aware section chunking; chunk_markdown with provenance.

    Same section walk (ATX heading stack) and same paragraph split
    ("\n".join(body).split("\n\n"), keeping leading-"\n" pieces) as
    chunk_markdown, tracking each piece's span in the original file. Body
    lines are consecutive file lines, so the joined section text is an exact
    file substring and piece offsets map straight through. Neither book file
    has \r, so line arithmetic is exact.
    """
    sections: list[tuple[str, list[tuple[str, int, int]]]] = []  # (path, [(line, char_start, lineno)])
    stack: list[tuple[int, str]] = []  # (level, title)
    body: list[tuple[str, int, int]] = []

    def flush() -> None:
        path = " > ".join(title for _, title in stack)
        sections.append((path, list(body)))
        body.clear()

    pos = 0
    for lineno, line in enumerate(text.split("\n"), 1):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append((line, pos, lineno))
        pos += len(line) + 1
    flush()

    chunks: list[MarkdownChunk] = []
    for path, records in sections:
        if not records:
            continue
        sec_start = records[0][1]  # joined section text == text[sec_start:...]
        line_at = {r[1]: r[2] for r in records}  # char_start -> lineno
        paras: list[tuple[str, int, int, int]] = []  # (text, char_start, char_end, line_start)
        piece_pos = 0
        for piece in "\n".join(r[0] for r in records).split("\n\n"):
            if piece.strip():
                cs = sec_start + piece_pos  # pieces start at line starts
                paras.append((piece, cs, cs + len(piece), line_at[cs]))
            piece_pos += len(piece) + 2  # skip the "\n\n" separator
        for content, char_start, char_end, line_start in _merge_paras(paras, max_chars):
            chunks.append(MarkdownChunk(content, path or None, char_start, char_end, line_start))
    return chunks


def chunk_markdown(text: str, max_chars: int = 800) -> list[tuple[str, str]]:
    """Section-aware chunking. Returns list of (content, heading_path).

    ATX headings build a heading stack; body lines accumulate under the
    current heading; each section is split on blank lines and greedily
    merged up to max_chars. Docs with no headings are one section.
    """
    return [(c.content, c.heading_path) for c in chunk_markdown_metadata(text, max_chars)]