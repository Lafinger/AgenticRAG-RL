from __future__ import annotations

from pathlib import Path

from agentic_rag_rl.chunking import chunk_text_file


def test_chunk_text_file_generates_stable_novel_chunks(tmp_path: Path) -> None:
    source = tmp_path / "平凡的世界utf8.txt"
    source.write_text(
        "\r\n".join(
            [
                "孙少平在学校生活艰难。",
                "郝红梅也因为贫穷而自卑。",
                "",
                "双水村有东拉河和哭咽河。",
            ]
        ),
        encoding="utf-8",
    )

    chunks = chunk_text_file(source, chunk_chars=40, overlap_chars=0)

    assert chunks
    assert chunks[0].chunk_id == "corpus_chunkids_000001"
    assert chunks[0].title.startswith("平凡的世界")
    assert chunks[0].metadata["source_file"] == "平凡的世界utf8.txt"
    assert chunks[0].metadata["line_start"] == 1
    assert "孙少平" in chunks[0].metadata["character_aliases"]
    assert any("双水村" in chunk.metadata["character_aliases"] for chunk in chunks)
