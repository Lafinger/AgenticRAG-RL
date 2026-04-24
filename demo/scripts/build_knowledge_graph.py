from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_rl.io import load_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight knowledge graph from corpus aliases.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-graph", required=True)
    parser.add_argument("--output-embeddings", required=False)
    args = parser.parse_args()

    chunks = load_chunks(args.corpus)
    graph = {"nodes": [], "edges": []}
    seen_nodes: set[str] = set()

    for chunk in chunks:
        company = chunk.company or chunk.title
        if company not in seen_nodes:
            graph["nodes"].append({"id": company, "type": "company"})
            seen_nodes.add(company)
        for alias in chunk.metadata.get("aliases", []):
            if alias not in seen_nodes:
                graph["nodes"].append({"id": alias, "type": "alias"})
                seen_nodes.add(alias)
            graph["edges"].append({"source": company, "target": alias, "relation": "alias_of", "chunk_id": chunk.chunk_id})

    output = Path(args.output_graph)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_embeddings:
        Path(args.output_embeddings).write_text(json.dumps({"status": "lightweight-placeholder"}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"nodes={len(graph['nodes'])}")
    print(f"edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
