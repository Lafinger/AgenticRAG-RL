from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
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


def build_chunk_id(prefix: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{index:04d}_{digest}"


def chunk_pdf(pdf_path: str | Path, title: str | None = None, prefix: str | None = None) -> list[dict[str, str]]:
    pdf_path = Path(pdf_path)
    title = title or pdf_path.stem
    prefix = prefix or re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_") or "doc"
    text = read_pdf_text(pdf_path)
    chunks = split_into_chunks(text)
    records: list[dict[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
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


def write_jsonl(records: Iterable[dict[str, object]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

