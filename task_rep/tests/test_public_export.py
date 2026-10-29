"""Publication boundary: UI exports exclude internal run data and machine paths."""
from export_bundle import export_bundle


def test_export_keeps_cross_source_units_and_all_labels_without_internal_metadata():
    result = {"pool": [{"id": "f", "title": "Paper", "vocabulary": [{"label": "thin"}]}],
              "annotations": [["a", "f", "thin"], ["a", "f", "thick"], ["a", "f", "thin"]],
              "runner_state": {"private_prompt": "internal"}, "usage": {"private": 1}}
    kb = {"items": [{"id": name, "type": "step", "text": "Fold", "unit_id": "shared",
                     "source": {"video_id": video, "time_range": [0, 5]},
                     "metadata": {"path": "/private/input"}} for name, video in [("a", "v1"), ("b", "v2")]]}
    videos = [{"id": "v1", "url": "/media/demo.mp4", "title": "Synthetic tutorial", "frames": [{"path": "/private/frame"}]}]
    bundle = export_bundle(result, kb, videos)
    assert set(bundle) == {"context_schema", "facet_candidates", "labeled_dataset"}
    first, second = bundle["labeled_dataset"]
    assert first["url"] == "/media/demo.mp4" and second["url"] == ""
    assert first["pieces"][0]["unit_id"] == second["pieces"][0]["unit_id"] == "shared"
    assert first["pieces"][0]["labels"] == {"f": ["thin", "thick"]}
    assert "private" not in str(bundle) and "internal" not in str(bundle)
