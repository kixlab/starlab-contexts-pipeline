"""Export a saved pipeline result and knowledge base to the public UI format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_bundle(result: dict, knowledge_base: dict, dataset: list | None = None) -> dict:
    """Only publish UI fields: no model logs, prompts, paths, or run metadata."""
    facets = []
    for facet in result.get("pool", result.get("facets", [])):
        facets.append({
            "id": facet["id"], "type": facet.get("scope", facet.get("type", "local")),
            "title": facet["title"], "title_plural": facet.get("title_plural", facet["title"]),
            "definition": facet.get("definition", ""), "guidelines": facet.get("guidelines", []),
            "vocabulary": [{"label": value["label"], "definition": value.get("definition", "")}
                           for value in facet.get("vocabulary", [])],
        })
    facet_ids = {facet["id"] for facet in facets}
    labels = {}
    for item, facet, label in result.get("annotations", []):
        if facet in facet_ids:
            labels.setdefault(item, {}).setdefault(facet, []).append(label)
    sources = {video["id"]: video for video in dataset or []}
    videos = {}
    for item in knowledge_base["items"]:
        source = item["source"]
        video_id = source["video_id"]
        meta = sources.get(video_id, {})
        if video_id not in videos:
            videos[video_id] = {
                "id": video_id, "url": meta.get("url", ""),
                "title": meta.get("title", video_id), "original_title": meta.get("title", video_id),
                "content": "", "transcript": [], "pieces": [],
            }
        start, end = source.get("time_range") or (0, 0)
        piece = {
            "piece_id": item["id"], "unit_id": item.get("unit_id") or item["id"],
            "item_type": item["type"], "content_type": item.get("subtype") or item["type"],
            "start": start, "end": end, "content": item["text"],
            "labels": labels.get(item["id"], {}),
        }
        videos[video_id]["pieces"].append(piece)
    for video in videos.values():
        video["pieces"].sort(key=lambda piece: (piece["start"], piece["piece_id"]))
        video["content"] = " ".join(piece["content"] for piece in video["pieces"])
        video["transcript"] = [{"text": piece["content"], "start": piece["start"], "end": piece["end"]}
                               for piece in video["pieces"]]
    return {"context_schema": facets, "facet_candidates": facets, "labeled_dataset": list(videos.values())}


def load_knowledge_base(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return {"items": [json.loads(line) for line in path.read_text().splitlines() if line.strip()]}
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Saved result.json")
    parser.add_argument("--knowledge-base", required=True, type=Path, help="Knowledge-base JSON or cached units JSONL")
    parser.add_argument("--dataset", type=Path, help="Optional task dataset with video URLs and titles")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = export_bundle(json.loads(args.result.read_text()), load_knowledge_base(args.knowledge_base),
                           json.loads(args.dataset.read_text()) if args.dataset else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
