#!/usr/bin/env python3
"""Compare a legacy ``scripts/legacy_piper/preprocess_piper.py`` dataset with ``data`` output.

    python scripts/compare_piper_preprocessing.py LEGACY_DATASET NEW_DATASET

Requires identical parquet columns/values, episode/task/source manifests,
cleanup metrics, core info fields, and video frame counts.  Legacy
``modality.json`` entries must be preserved (the package may add keys such
as ``quaternion_policy``).  Videos are compared byte-for-byte and, if the
encoder output differs, by decoded frames.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CORE_INFO = ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames",
             "total_tasks", "total_videos", "chunks_size", "splits", "data_path", "video_path",
             "state_source", "action_source", "state_layout", "action_layout",
             "action_timestep_offset")
COLUMNS = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
           "task_index")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _subset(old, new, label: str) -> None:
    if isinstance(old, dict):
        assert isinstance(new, dict), label
        for key, value in old.items():
            assert key in new, f"{label}: missing {key}"
            _subset(value, new[key], f"{label}.{key}")
    else:
        assert old == new, f"{label}: {old!r} != {new!r}"


def compare(legacy: Path, new: Path) -> dict:
    old_info, new_info = _json(legacy / "meta/info.json"), _json(new / "meta/info.json")
    for key in CORE_INFO:
        assert old_info[key] == new_info[key], f"info.{key}: {old_info[key]!r} != {new_info[key]!r}"
    for key, feature in old_info["features"].items():
        assert feature["dtype"] == new_info["features"][key]["dtype"], key
        assert feature["shape"] == new_info["features"][key]["shape"], key
    for key in ("removed_frames", "config", "motion_source", "enabled", "method"):
        assert old_info["cleanup"][key] == new_info["cleanup"][key], f"cleanup.{key}"
    _subset(_json(legacy / "meta/modality.json"), _json(new / "meta/modality.json"), "modality")
    for name in ("episodes.jsonl", "tasks.jsonl", "source_episodes.jsonl"):
        assert (legacy / "meta" / name).read_text() == (new / "meta" / name).read_text(), name
    with (legacy / "meta/cleanup_metrics.csv").open() as a, (new / "meta/cleanup_metrics.csv").open() as b:
        assert list(csv.DictReader(a)) == list(csv.DictReader(b)), "cleanup_metrics.csv"
    rows = videos_identical = videos_decoded = 0
    max_pixel_difference = 0.0
    for old_parquet in sorted((legacy / "data").rglob("*.parquet")):
        relative = old_parquet.relative_to(legacy)
        old_table, new_table = pq.read_table(old_parquet), pq.read_table(new / relative)
        for column in COLUMNS:
            a = np.asarray(old_table[column].to_pylist())
            b = np.asarray(new_table[column].to_pylist())
            assert a.dtype == b.dtype and np.array_equal(a, b), f"{relative}: {column} differs"
        rows += len(old_table)
    for old_video in sorted((legacy / "videos").rglob("*.mp4")):
        relative = old_video.relative_to(legacy)
        new_video = new / relative
        if sha256(old_video.read_bytes()).digest() == sha256(new_video.read_bytes()).digest():
            videos_identical += 1
            continue
        import cv2

        a, b = cv2.VideoCapture(str(old_video)), cv2.VideoCapture(str(new_video))
        try:
            assert a.get(cv2.CAP_PROP_FRAME_COUNT) == b.get(cv2.CAP_PROP_FRAME_COUNT), relative
            while True:
                ok_a, frame_a = a.read()
                ok_b, frame_b = b.read()
                assert ok_a == ok_b, relative
                if not ok_a:
                    break
                max_pixel_difference = max(max_pixel_difference, float(
                    np.abs(frame_a.astype(int) - frame_b.astype(int)).max()))
        finally:
            a.release()
            b.release()
        videos_decoded += 1
    return {"passed": True, "rows": rows, "videos_byte_identical": videos_identical,
            "videos_compared_by_frames": videos_decoded,
            "max_pixel_difference": max_pixel_difference}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.legacy, args.new), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
