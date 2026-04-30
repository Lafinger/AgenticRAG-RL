from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.chunking import chunk_text_file, chunk_text_file_to_jsonl


def test_chunk_text_file_generates_stable_novel_chapter_chunks(tmp_path: Path) -> None:
    source = tmp_path / "平凡的世界-路遥.txt"
    source.write_text(
        "\r\n".join(
            [
                "第一部 第一章",
                "孙少平在学校生活艰难。",
                "郝红梅也因为贫穷而自卑。",
                "",
                "第二章",
                "孙兰花住在双水村附近。",
            ]
        ),
        encoding="utf-8",
        newline="",
    )

    chunks = chunk_text_file(source)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "corpus_chunkids_000001"
    assert chunks[0].title == "平凡的世界 第一部 第一章"
    assert chunks[0].metadata["source_file"] == "平凡的世界-路遥.txt"
    assert chunks[0].metadata["line_start"] == 1
    assert chunks[0].metadata["line_end"] == 3
    assert chunks[0].metadata["chunk_type"] == "chapter"
    assert chunks[0].metadata["part_title"] == "第一部"
    assert chunks[0].metadata["chapter_title"] == "第一章"
    assert "孙少平" in chunks[0].metadata["character_aliases"]
    assert chunks[1].title == "平凡的世界 第一部 第二章"
    assert "孙兰花" in chunks[1].metadata["character_aliases"]
    assert all("双水村" not in chunk.metadata["character_aliases"] for chunk in chunks)


def test_chunk_text_file_to_jsonl_keeps_readable_chinese_and_omits_company(tmp_path: Path) -> None:
    source = tmp_path / "平凡的世界-路遥.txt"
    output = tmp_path / "corpus.jsonl"
    source.write_text("第一部 第一章\r\n孙少平在学校生活艰难。\r\n", encoding="utf-8", newline="")

    chunk_text_file_to_jsonl(source, output)

    text = output.read_text(encoding="utf-8")
    assert "孙少平" in text
    assert "corpus_chunkids_000001" in text
    assert '"chapter_title": "第一章"' in text
    assert '"company"' not in text
