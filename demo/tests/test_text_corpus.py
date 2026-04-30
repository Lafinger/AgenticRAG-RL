from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_rl.chunking import chunk_text_dir, chunk_text_dir_to_jsonl, chunk_text_file


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\r\n".join(lines), encoding="utf-8", newline="")


def test_chunk_text_dir_parses_multiple_jinyong_sources(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "金庸-天龙八部.txt",
        [
            "一 青衫磊落险峰行",
            "乔峰来到少室山，段誉也在山道上相候。",
            "",
            "后记",
            "萧峰的故事后来仍被江湖人物谈起。",
        ],
    )
    _write_text(
        tmp_path / "金庸-侠客行.txt",
        [
            "第一回 烧饼馅子",
            "石破天拿着烧饼，谢烟客坐在一旁。",
        ],
    )

    chunks = chunk_text_dir(tmp_path, chunk_chars=80, overlap_chars=10, min_chars=10)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    assert {chunk.metadata["source_file"] for chunk in chunks} == {"金庸-天龙八部.txt", "金庸-侠客行.txt"}
    assert "tlbb_0001" in by_id
    assert "xkx_0001" in by_id
    assert by_id["tlbb_0001"].title == "天龙八部"
    assert by_id["tlbb_0001"].section == "一 青衫磊落险峰行"
    assert by_id["tlbb_0001"].pages == []
    assert by_id["tlbb_0001"].metadata["author"] == "金庸"
    assert by_id["tlbb_0001"].metadata["section_index"] == 1
    assert "乔峰" in by_id["tlbb_0001"].metadata["character_aliases"]


def test_chunk_text_file_supports_sectionless_short_story(tmp_path: Path) -> None:
    source = tmp_path / "金庸-越女剑.txt"
    _write_text(source, ["越女剑", "阿青见到范蠡，西施也在越国宫中。"])

    chunks = chunk_text_file(source, chunk_chars=80, overlap_chars=10, min_chars=10)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "yue_0001"
    assert chunks[0].title == "越女剑"
    assert chunks[0].section == "越女剑"
    assert {"阿青", "范蠡", "西施"} <= set(chunks[0].metadata["character_aliases"])


def test_chunk_text_dir_to_jsonl_writes_root_schema(tmp_path: Path) -> None:
    source_dir = tmp_path / "original_data"
    source_dir.mkdir()
    output = tmp_path / "corpus.jsonl"
    _write_text(source_dir / "金庸-笑傲江湖.txt", ["一 灭门", "令狐冲听见岳不群说起林平之。"])

    chunk_text_dir_to_jsonl(source_dir, output, chunk_chars=80, overlap_chars=10, min_chars=10)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["chunk_id"] == "xajh_0001"
    assert records[0]["title"] == "笑傲江湖"
    assert records[0]["section"] == "一 灭门"
    assert records[0]["pages"] == []
    assert records[0]["metadata"]["novel_title"] == "笑傲江湖"
    assert '"company"' not in output.read_text(encoding="utf-8")
