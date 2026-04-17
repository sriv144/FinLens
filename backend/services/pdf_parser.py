import re

import fitz  # PyMuPDF

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE


INDEX_VERSION = "textbook_v2"

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_CHAPTER_HEADING_RE = re.compile(
    r"(?:^|\n)(?:chapter\s+|c\s*h\s*a\s*p\s*t\s*e\s*r\s*\n*)(\d+)(?:\s*\n+|\s+)([A-Za-z][A-Za-z0-9 ,:/()\-]{2,})",
    flags=re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)+)\s+(?P<title>[A-Z][A-Za-z0-9 ,:/()\-]{2,})$")
_CONTENTS_SECTION_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)+)\s+(?P<title>.+?)\s+(?P<page>\d+)$"
)
_CHAPTER_SUMMARY_RE = re.compile(
    r"\bChapter\s+(?P<number>\d+)\s+(?P<body>(?:is|covers?|overviews?|introduces?)\s+.+?)(?:\.|;)",
    flags=re.IGNORECASE,
)


def extract_text_from_pdf(filepath: str) -> list[dict]:
    """Return a list of {page_num, text} dictionaries."""
    doc = fitz.open(filepath)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_num": page_num, "text": text})
    doc.close()
    return pages


def build_document_index(pages: list[dict]) -> dict:
    """Build a textbook-oriented index with structure and passage units."""
    structure_units: list[dict] = []
    passage_units: list[dict] = []
    structure_id = 0
    passage_id = 0
    global_offset = 0
    current_chapter: int | None = None
    current_section: str | None = None

    for page in pages:
        page_num = page["page_num"]
        page_text = page["text"]
        lines = _normalize_lines(page_text)
        normalized_page = _normalize_page_text(page_text)

        chapter_heading = _extract_chapter_heading(lines)
        if chapter_heading:
            current_chapter = chapter_heading["chapter_number"]
            current_section = None
            structure_units.append(
                _build_structure_unit(
                    unit_id=structure_id,
                    page_num=page_num,
                    char_start=global_offset,
                    evidence_type="chapter_heading",
                    quote=chapter_heading["quote"],
                    chapter_number=chapter_heading["chapter_number"],
                    section_label=chapter_heading["title"],
                )
            )
            structure_id += 1

        section_headings = _extract_section_headings(lines, current_chapter)
        if section_headings:
            current_chapter = section_headings[0]["chapter_number"] or current_chapter
            for section in section_headings:
                structure_units.append(
                    _build_structure_unit(
                        unit_id=structure_id,
                        page_num=page_num,
                        char_start=global_offset + section["char_start"],
                        evidence_type="section_heading",
                        quote=section["quote"],
                        chapter_number=section["chapter_number"],
                        section_label=section["section_label"],
                    )
                )
                structure_id += 1

        if _is_contents_page(lines):
            for entry in _extract_contents_entries(lines, global_offset):
                structure_units.append(
                    _build_structure_unit(
                        unit_id=structure_id,
                        page_num=page_num,
                        char_start=entry["char_start"],
                        evidence_type="contents_entry",
                        quote=entry["quote"],
                        chapter_number=entry.get("chapter_number"),
                        section_label=entry.get("section_label"),
                    )
                )
                structure_id += 1

        for summary in _extract_chapter_summaries(normalized_page):
            structure_units.append(
                _build_structure_unit(
                    unit_id=structure_id,
                    page_num=page_num,
                    char_start=global_offset,
                    evidence_type="chapter_summary",
                    quote=summary["quote"],
                    chapter_number=summary["chapter_number"],
                    section_label=summary.get("section_label"),
                )
            )
            structure_id += 1

        page_passages, current_section, passage_id = _build_passage_units_for_page(
            lines=lines,
            page_num=page_num,
            page_offset=global_offset,
            start_unit_id=passage_id,
            current_chapter=current_chapter,
            current_section=current_section,
        )
        passage_units.extend(page_passages)
        global_offset += len(page_text) + 1

    return {
        "index_version": INDEX_VERSION,
        "page_count": len(pages),
        "structure_units": structure_units,
        "passage_units": passage_units,
    }


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Backward-compatible helper that returns passage units."""
    return build_document_index(pages)["passage_units"]


def _normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_page_text(text: str) -> str:
    normalized = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    normalized = re.sub(r"\s*\n\s*", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _build_structure_unit(
    unit_id: int,
    page_num: int,
    char_start: int,
    evidence_type: str,
    quote: str,
    chapter_number: int | None = None,
    section_label: str | None = None,
) -> dict:
    return {
        "unit_id": unit_id,
        "text": quote.strip(),
        "page_num": page_num,
        "char_start": char_start,
        "evidence_type": evidence_type,
        "chapter_number": chapter_number,
        "section_label": section_label,
    }


def _extract_chapter_heading(lines: list[str]) -> dict | None:
    if not lines:
        return None

    joined = "\n".join(lines[:8])
    match = _CHAPTER_HEADING_RE.search(joined)
    if match:
        chapter_number = int(match.group(1))
        title = match.group(2).strip()
        return {
            "chapter_number": chapter_number,
            "title": title,
            "quote": f"Chapter {chapter_number}\n{title}",
        }

    if len(lines) >= 3 and lines[0].replace(" ", "").lower() == "chapter" and lines[1].isdigit():
        chapter_number = int(lines[1])
        title = lines[2]
        return {
            "chapter_number": chapter_number,
            "title": title,
            "quote": f"Chapter {chapter_number}\n{title}",
        }
    return None


def _extract_section_headings(lines: list[str], current_chapter: int | None) -> list[dict]:
    headings = []
    running_pos = 0
    for line in lines:
        match = _SECTION_RE.match(line)
        if match:
            section_number = match.group("number")
            chapter_number = int(section_number.split(".")[0])
            title = match.group("title").strip()
            headings.append(
                {
                    "quote": f"{section_number} {title}",
                    "section_label": section_number,
                    "chapter_number": chapter_number or current_chapter,
                    "char_start": running_pos,
                }
            )
        running_pos += len(line) + 1
    return headings


def _is_contents_page(lines: list[str]) -> bool:
    if not lines:
        return False
    header = " ".join(line.lower() for line in lines[:6])
    if "contents" not in header and "table of contents" not in header:
        return False
    digit_lines = sum(1 for line in lines if re.match(r"^\d+(?:\.\d+)?$", line))
    return digit_lines >= 4


def _extract_contents_entries(lines: list[str], page_offset: int) -> list[dict]:
    entries = []
    running_pos = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.lower() in {"contents", "table of contents", "preface"}:
            running_pos += len(line) + 1
            index += 1
            continue

        section_match = _CONTENTS_SECTION_RE.match(line)
        if section_match:
            section_number = section_match.group("number")
            title = section_match.group("title").strip()
            entries.append(
                {
                    "quote": line,
                    "char_start": page_offset + running_pos,
                    "chapter_number": int(section_number.split(".")[0]),
                    "section_label": section_number,
                }
            )
            running_pos += len(line) + 1
            index += 1
            continue

        chapter_match = re.match(r"^(\d+)$", line)
        if chapter_match and index + 2 < len(lines):
            chapter_number = int(chapter_match.group(1))
            title_parts = []
            lookahead = index + 1
            while lookahead < len(lines) and not re.match(r"^\d+$", lines[lookahead]):
                if lines[lookahead].lower() not in {"contents", "table of contents"}:
                    title_parts.append(lines[lookahead])
                lookahead += 1
            if title_parts and lookahead < len(lines):
                quote_lines = [line, *title_parts, lines[lookahead]]
                entries.append(
                    {
                        "quote": "\n".join(quote_lines),
                        "char_start": page_offset + running_pos,
                        "chapter_number": chapter_number,
                        "section_label": " ".join(title_parts),
                    }
                )
                consumed = quote_lines
                running_pos += sum(len(item) + 1 for item in consumed)
                index = lookahead + 1
                continue

        running_pos += len(line) + 1
        index += 1

    return entries


def _extract_chapter_summaries(normalized_page: str) -> list[dict]:
    summaries = []
    for match in _CHAPTER_SUMMARY_RE.finditer(normalized_page):
        chapter_number = int(match.group("number"))
        body = match.group("body").strip()
        summaries.append(
            {
                "chapter_number": chapter_number,
                "quote": f"Chapter {chapter_number} {body}.",
            }
        )
    return summaries


def _build_passage_units_for_page(
    lines: list[str],
    page_num: int,
    page_offset: int,
    start_unit_id: int,
    current_chapter: int | None,
    current_section: str | None,
) -> tuple[list[dict], str | None, int]:
    if not lines:
        return [], current_section, start_unit_id

    passages = []
    segments = []
    segment_lines: list[str] = []
    segment_offset = 0
    segment_section = current_section
    running_pos = 0

    for line in lines:
        section_match = _SECTION_RE.match(line)
        if section_match:
            if segment_lines:
                segments.append((segment_offset, "\n".join(segment_lines), current_chapter, segment_section))
            segment_lines = [line]
            segment_offset = running_pos
            segment_section = section_match.group("number")
            current_chapter = int(segment_section.split(".")[0])
        else:
            if not segment_lines:
                segment_offset = running_pos
            segment_lines.append(line)
        running_pos += len(line) + 1

    if segment_lines:
        segments.append((segment_offset, "\n".join(segment_lines), current_chapter, segment_section))

    unit_id = start_unit_id
    last_section = current_section
    for segment_offset, segment_text, segment_chapter, segment_section in segments:
        last_section = segment_section or last_section
        raw_chunks = _recursive_split(segment_text, CHUNK_SIZE)
        chunks = _add_overlap(raw_chunks, segment_text, CHUNK_OVERLAP)
        for local_offset, chunk_text in chunks:
            text = chunk_text.strip()
            if not text:
                continue
            passages.append(
                {
                    "unit_id": unit_id,
                    "text": text,
                    "page_num": page_num,
                    "char_start": page_offset + segment_offset + local_offset,
                    "chapter_number": segment_chapter,
                    "section_label": segment_section,
                }
            )
            unit_id += 1

    return passages, last_section, unit_id


def _recursive_split(text: str, chunk_size: int) -> list[tuple[int, str]]:
    """Split text recursively while trying to preserve natural boundaries."""
    if len(text) <= chunk_size:
        return [(0, text)] if text.strip() else []

    for sep in _SEPARATORS:
        if sep == "":
            return [(i, text[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: list[tuple[int, str]] = []
        current = ""
        current_start = 0
        running_offset = 0

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                if not current:
                    current_start = running_offset
                current = candidate
            else:
                if current:
                    chunks.append((current_start, current))
                if len(part) > chunk_size:
                    for sub_off, sub_text in _recursive_split(part, chunk_size):
                        chunks.append((running_offset + sub_off, sub_text))
                    current = ""
                    current_start = running_offset + len(part) + len(sep)
                else:
                    current = part
                    current_start = running_offset
            running_offset += len(part) + len(sep)

        if current:
            chunks.append((current_start, current))

        return chunks

    return [(0, text)]


def _add_overlap(chunks: list[tuple[int, str]], full_text: str, overlap: int) -> list[tuple[int, str]]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for char_start, text in chunks[1:]:
        prefix_start = max(0, char_start - overlap)
        prefix = full_text[prefix_start:char_start]
        new_text = prefix + text if prefix.strip() else text
        result.append((prefix_start if prefix.strip() else char_start, new_text))
    return result
