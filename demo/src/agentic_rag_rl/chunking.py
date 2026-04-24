from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from pathlib import Path

from pypdf import PdfReader

from .io import write_jsonl
from .types import Chunk


NOVEL_ALIASES = [
    "孙少平",
    "郝红梅",
    "双水村",
    "孙少安",
    "田润叶",
    "田晓霞",
    "金波",
    "孙玉厚",
    "王满银",
    "兰花",
    "田福堂",
    "顾养民",
    "侯玉英",
]

INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


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


def _paragraph_records(text: str) -> list[tuple[str, int, int]]:
    records: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    start_line = 1
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            if buffer:
                records.append(("\n".join(buffer), start_line, line_no - 1))
                buffer = []
            continue
        if not buffer:
            start_line = line_no
        buffer.append(stripped)
    if buffer:
        records.append(("\n".join(buffer), start_line, len(lines)))
    return records


def chunk_text_file(
    input_path: str | Path,
    title: str = "平凡的世界",
    prefix: str = "corpus_chunkids",
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> list[Chunk]:
    source = Path(input_path)
    text = source.read_text(encoding="utf-8-sig")
    paragraphs = _paragraph_records(text)
    chunks: list[Chunk] = []
    current_text = ""
    current_start = 1
    current_end = 1

    def flush() -> None:
        nonlocal current_text, current_start, current_end
        if not current_text.strip():
            return
        index = len(chunks) + 1
        body = normalize_text(current_text)
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}_{index:06d}",
                title=f"{title} 段落 {index}",
                text=body,
                company="",
                metadata={
                    "source_file": source.name,
                    "line_start": current_start,
                    "line_end": current_end,
                    "character_aliases": _detect_aliases(body),
                },
            )
        )
        if overlap_chars > 0 and len(body) > overlap_chars:
            current_text = body[-overlap_chars:]
            current_start = current_end
        else:
            current_text = ""

    for paragraph, line_start, line_end in paragraphs:
        candidate = paragraph if not current_text else f"{current_text}\n\n{paragraph}"
        if len(candidate) <= chunk_chars:
            if not current_text:
                current_start = line_start
            current_text = candidate
            current_end = line_end
            continue

        flush()
        current_text = paragraph
        current_start = line_start
        current_end = line_end

        while len(current_text) > chunk_chars:
            piece = current_text[:chunk_chars]
            index = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{prefix}_{index:06d}",
                    title=f"{title} 段落 {index}",
                    text=normalize_text(piece),
                    company="",
                    metadata={
                        "source_file": source.name,
                        "line_start": current_start,
                        "line_end": current_end,
                        "character_aliases": _detect_aliases(piece),
                    },
                )
            )
            current_text = current_text[max(chunk_chars - overlap_chars, 1) :]

    flush()
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
    chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> None:
    records = []
    for chunk in chunk_text_file(input_path, title=title, prefix=prefix, chunk_chars=chunk_chars, overlap_chars=overlap_chars):
        record = asdict(chunk)
        record.pop("company", None)
        records.append(record)
    write_jsonl(records, output_path)
