from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .io import write_jsonl
from .types import Chunk


NOVEL_ALIASES = [
    "乔峰",
    "萧峰",
    "段誉",
    "虚竹",
    "王语嫣",
    "阿朱",
    "阿紫",
    "慕容复",
    "鸠摩智",
    "天山童姥",
    "石破天",
    "石中玉",
    "丁珰",
    "闵柔",
    "谢烟客",
    "令狐冲",
    "任盈盈",
    "岳不群",
    "林平之",
    "东方不败",
    "任我行",
    "张无忌",
    "赵敏",
    "周芷若",
    "小昭",
    "殷离",
    "谢逊",
    "张三丰",
    "郭襄",
    "杨过",
    "小龙女",
    "阿青",
    "范蠡",
    "西施",
    "勾践",
]

SOURCE_PREFIXES = {
    "天龙八部": "tlbb",
    "侠客行": "xkx",
    "笑傲江湖": "xajh",
    "倚天屠龙记": "yttlj",
    "越女剑": "yue",
}

INVISIBLE_UNICODE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
CHINESE_NUMERAL = r"[一二三四五六七八九十百千万零〇两]+"
SECTION_PATTERNS = [
    re.compile(rf"^(第{CHINESE_NUMERAL}回)\s+(.{{1,60}})$"),
    re.compile(rf"^(第[\d０-９]+回)\s+(.{{1,60}})$"),
    re.compile(rf"^(第{CHINESE_NUMERAL}章)(?:\s+(.{{1,60}}))?$"),
    re.compile(rf"^({CHINESE_NUMERAL})\s+(.{{1,60}})$"),
]
SPECIAL_SECTION_PATTERN = re.compile(r"^(后记|楔子|引子|尾声|序|序章|附录(?:\s+.{1,60})?)$")
SENTENCE_END_PATTERN = re.compile(r"[。！？；：，,.!?;:]$")


@dataclass(slots=True)
class SectionRecord:
    title: str
    heading: str
    line_start: int
    line_end: int
    lines: list[tuple[int, str]]


@dataclass(slots=True)
class TextChunkRecord:
    text: str
    line_start: int
    line_end: int


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


def _derive_novel_title(source: Path) -> str:
    stem = source.stem
    if stem.startswith("金庸-"):
        return stem.split("-", 1)[1]
    return stem


def _derive_prefix(source: Path) -> str:
    novel_title = _derive_novel_title(source)
    if novel_title in SOURCE_PREFIXES:
        return SOURCE_PREFIXES[novel_title]
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", source.stem.lower()).strip("_")
    if ascii_slug:
        return ascii_slug
    digest = hashlib.sha1(source.stem.encode("utf-8")).hexdigest()[:8]
    return f"doc_{digest}"


def _parse_section_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or SENTENCE_END_PATTERN.search(stripped):
        return None
    special = SPECIAL_SECTION_PATTERN.fullmatch(stripped)
    if special:
        return stripped
    for pattern in SECTION_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match:
            return stripped
    return None


def _section_records(text: str, fallback_title: str) -> list[SectionRecord]:
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_found = False
    records: list[SectionRecord] = []
    current_title = "序"
    current_heading = "序"
    current_lines: list[tuple[int, str]] = []
    current_start: int | None = None

    def flush(line_end: int) -> None:
        nonlocal current_lines, current_start
        if current_start is None:
            return
        body = normalize_text("\n".join(line for _, line in current_lines))
        if body:
            records.append(
                SectionRecord(
                    title=current_title,
                    heading=current_heading,
                    line_start=current_start,
                    line_end=line_end,
                    lines=current_lines,
                )
            )
        current_lines = []
        current_start = None

    for line_no, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        section_heading = _parse_section_heading(stripped)
        if section_heading:
            heading_found = True
            flush(line_no - 1)
            current_title = section_heading
            current_heading = section_heading
            current_start = line_no
            current_lines = [(line_no, stripped)]
            continue

        if stripped and current_start is None:
            current_start = line_no
        if current_start is not None:
            current_lines.append((line_no, stripped))

    flush(len(raw_lines))
    if heading_found:
        return records

    body_lines = [(idx, line.strip()) for idx, line in enumerate(raw_lines, start=1)]
    body = normalize_text("\n".join(line for _, line in body_lines))
    if not body:
        return []
    first_line = next((idx for idx, line in body_lines if line), 1)
    last_line = next((idx for idx, line in reversed(body_lines) if line), len(raw_lines))
    return [
        SectionRecord(
            title=fallback_title,
            heading=fallback_title,
            line_start=first_line,
            line_end=last_line,
            lines=body_lines,
        )
    ]


def _paragraph_records(lines: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
    paragraphs: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    start_line = 1
    end_line = 1
    for line_no, line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                paragraphs.append(("\n".join(buffer), start_line, end_line))
                buffer = []
            continue
        if not buffer:
            start_line = line_no
        end_line = line_no
        buffer.append(stripped)
    if buffer:
        paragraphs.append(("\n".join(buffer), start_line, end_line))
    return paragraphs


def _chunk_section(
    section: SectionRecord,
    *,
    chunk_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[TextChunkRecord]:
    chunks: list[TextChunkRecord] = []
    current_text = ""
    current_start = section.line_start
    current_end = section.line_start

    def append_chunk(raw_text: str, line_start: int, line_end: int) -> None:
        body = normalize_text(raw_text)
        if len(body) >= min_chars:
            chunks.append(TextChunkRecord(text=body, line_start=line_start, line_end=line_end))

    def flush(keep_overlap: bool = True) -> None:
        nonlocal current_text, current_start, current_end
        body = normalize_text(current_text)
        if body:
            append_chunk(body, current_start, current_end)
        if keep_overlap and overlap_chars > 0 and len(body) > overlap_chars:
            current_text = body[-overlap_chars:]
            current_start = current_end
        else:
            current_text = ""

    for paragraph, line_start, line_end in _paragraph_records(section.lines):
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
            append_chunk(piece, current_start, current_end)
            current_text = current_text[max(chunk_chars - overlap_chars, 1) :]

    flush(keep_overlap=False)
    return chunks


def chunk_text_file(
    input_path: str | Path,
    title: str | None = None,
    prefix: str | None = None,
    *,
    chunk_chars: int = 500,
    overlap_chars: int = 50,
    min_chars: int = 50,
    author: str = "金庸",
) -> list[Chunk]:
    source = Path(input_path)
    novel_title = title or _derive_novel_title(source)
    chunk_prefix = prefix or _derive_prefix(source)
    text = source.read_text(encoding="utf-8-sig")
    chunks: list[Chunk] = []

    for section_index, section in enumerate(_section_records(text, novel_title), start=1):
        section_chunks = _chunk_section(
            section,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
        )
        for chunk_index_in_section, chunk_record in enumerate(section_chunks, start=1):
            chunk_index = len(chunks) + 1
            body = chunk_record.text
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_prefix}_{chunk_index:04d}",
                    title=novel_title,
                    text=body,
                    company="",
                    metadata={
                        "source_file": source.name,
                        "novel_title": novel_title,
                        "author": author,
                        "line_start": chunk_record.line_start,
                        "line_end": chunk_record.line_end,
                        "section_index": section_index,
                        "chunk_index_in_section": chunk_index_in_section,
                        "section_heading": section.heading,
                        "character_aliases": _detect_aliases(body),
                    },
                    pages=[],
                    section=section.title,
                )
            )
    return chunks


def chunk_text_files(
    input_paths: Iterable[str | Path],
    *,
    chunk_chars: int = 500,
    overlap_chars: int = 50,
    min_chars: int = 50,
    author: str = "金庸",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for input_path in sorted((Path(path) for path in input_paths), key=lambda item: item.name):
        chunks.extend(
            chunk_text_file(
                input_path,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
                min_chars=min_chars,
                author=author,
            )
        )
    return chunks


def chunk_text_dir(
    input_dir: str | Path,
    *,
    pattern: str = "*.txt",
    chunk_chars: int = 500,
    overlap_chars: int = 50,
    min_chars: int = 50,
    author: str = "金庸",
) -> list[Chunk]:
    source_dir = Path(input_dir)
    input_paths = sorted(source_dir.glob(pattern), key=lambda item: item.name)
    return chunk_text_files(
        input_paths,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        author=author,
    )


def _chunk_to_json_record(chunk: Chunk) -> dict[str, object]:
    record = asdict(chunk)
    record.pop("company", None)
    return record


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
    title: str | None = None,
    prefix: str | None = None,
    *,
    chunk_chars: int = 500,
    overlap_chars: int = 50,
    min_chars: int = 50,
    author: str = "金庸",
) -> None:
    records = [
        _chunk_to_json_record(chunk)
        for chunk in chunk_text_file(
            input_path,
            title=title,
            prefix=prefix,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            author=author,
        )
    ]
    write_jsonl(records, output_path)


def chunk_text_dir_to_jsonl(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    pattern: str = "*.txt",
    chunk_chars: int = 500,
    overlap_chars: int = 50,
    min_chars: int = 50,
    author: str = "金庸",
) -> None:
    records = [
        _chunk_to_json_record(chunk)
        for chunk in chunk_text_dir(
            input_dir,
            pattern=pattern,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
            min_chars=min_chars,
            author=author,
        )
    ]
    write_jsonl(records, output_path)
