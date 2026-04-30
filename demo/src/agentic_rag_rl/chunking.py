from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path

from pypdf import PdfReader

from .io import write_jsonl
from .types import Chunk


NOVEL_ALIASES = [
    "孙玉厚",
    "孙玉亭",
    "孙少平",
    "孙少安",
    "孙兰花",
    "孙兰香",
    "贺秀莲",
    "贺凤英",
    "贺耀宗",
    "王满银",
    "王世才",
    "王明明",
    "田福军",
    "田福堂",
    "田晓霞",
    "田润叶",
    "田润生",
    "李向前",
    "金波",
    "金秀",
    "金俊武",
    "金光亮",
    "金俊海",
    "郝红梅",
    "顾养民",
    "侯玉英",
    "杜丽丽",
    "武惠良",
    "吴仲平",
    "惠英",
    "小翠",
    "马顺",
    "胡老板",
    "贾有财",
]

INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
CHAPTER_HEADING_PATTERN = re.compile(
    r"^(?:(第[一二三四五六七八九十百千万零〇两\d０-９]+部)\s+)?"
    r"(第[一二三四五六七八九十百千万零〇两\d０-９]+章)$"
)


def normalize_text(text: str) -> str:
    text = INVISIBLE_UNICODE_PATTERN.sub("", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text("\n\n".join(pages))


def split_into_chunks(text: str, chunk_chars: int = 600, overlap_chars: int = 80) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in normalize_text(text).split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = start + chunk_chars
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start = max(end - overlap_chars, start + 1)
        current = ""

    if current:
        chunks.append(current)
    return chunks


def _detect_aliases(text: str) -> list[str]:
    return [alias for alias in NOVEL_ALIASES if alias in text]


def _chapter_records(text: str) -> list[tuple[str, str, str, str, int, int]]:
    records: list[tuple[str, str, str, str, int, int]] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    current_part = ""
    current_chapter = ""
    current_heading = ""
    current_lines: list[str] = []
    current_start = 1
    current_end = 1

    def flush() -> None:
        nonlocal current_heading, current_chapter, current_lines, current_start, current_end
        if not current_heading:
            return
        body = normalize_text("\n".join(current_lines))
        if body:
            records.append((current_part, current_chapter, current_heading, body, current_start, current_end))
        current_heading = ""
        current_chapter = ""
        current_lines = []

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        match = CHAPTER_HEADING_PATTERN.match(stripped)
        if match:
            flush()
            if match.group(1):
                current_part = match.group(1)
            current_chapter = match.group(2)
            current_heading = stripped
            current_lines = [stripped]
            current_start = line_no
            current_end = line_no
            continue

        if current_heading:
            current_lines.append(stripped)
            if stripped:
                current_end = line_no

    flush()
    if records:
        return records

    body = normalize_text(text)
    if not body:
        return []
    return [("", "全文", "全文", body, 1, len(lines))]


def chunk_text_file(
    input_path: str | Path,
    title: str = "平凡的世界",
    prefix: str = "corpus_chunkids",
) -> list[Chunk]:
    source = Path(input_path)
    text = source.read_text(encoding="utf-8-sig")
    chunks: list[Chunk] = []
    for index, (part_title, chapter_title, heading, body, line_start, line_end) in enumerate(_chapter_records(text), start=1):
        title_parts = [title]
        if part_title:
            title_parts.append(part_title)
        if chapter_title and chapter_title != "全文":
            title_parts.append(chapter_title)
        elif chapter_title == "全文":
            title_parts.append(chapter_title)
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}_{index:06d}",
                title=" ".join(title_parts),
                text=body,
                company="",
                metadata={
                    "source_file": source.name,
                    "line_start": line_start,
                    "line_end": line_end,
                    "chunk_type": "chapter",
                    "chapter_index": index,
                    "part_title": part_title,
                    "chapter_title": chapter_title,
                    "chapter_heading": heading,
                    "character_aliases": _detect_aliases(body),
                },
            )
        )
    return chunks


def build_chunk_id(prefix: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{index:04d}_{digest}"


def chunk_pdf(pdf_path: str | Path, title: str | None = None, prefix: str | None = None) -> list[dict[str, object]]:
    pdf = Path(pdf_path)
    title = title or pdf.stem
    prefix = prefix or re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()).strip("_") or "doc"
    text = read_pdf_text(pdf)
    records: list[dict[str, object]] = []
    for index, chunk in enumerate(split_into_chunks(text), start=1):
        records.append(
            {
                "chunk_id": build_chunk_id(prefix, index, chunk),
                "title": title,
                "text": chunk,
                "company": "",
                "metadata": {},
            }
        )
    return records


def chunk_pdf_to_jsonl(pdf_path: str | Path, output_path: str | Path, title: str | None = None, prefix: str | None = None) -> None:
    write_jsonl(chunk_pdf(pdf_path, title=title, prefix=prefix), output_path)


def chunk_text_file_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    title: str = "平凡的世界",
    prefix: str = "corpus_chunkids",
) -> None:
    records = []
    for chunk in chunk_text_file(input_path, title=title, prefix=prefix):
        record = asdict(chunk)
        record.pop("company", None)
        records.append(record)
    write_jsonl(records, output_path)
