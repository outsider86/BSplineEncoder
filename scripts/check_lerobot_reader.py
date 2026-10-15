#!/usr/bin/env python3
"""Read an exported v2.1 dataset with an installed LeRobot (>= 0.4, v3.0 reader).

    HF_HUB_OFFLINE=1 python scripts/check_lerobot_reader.py DATASET [--frames 3]

LeRobot 0.4 reads v3.0 datasets, so the check runs LeRobot's own
``convert_dataset_v21_to_v30`` on a temporary COPY (the input is never
modified; nothing is pushed), then loads it with ``LeRobotDataset`` and
compares every row's state/action/indices and a few decoded video frames with
the original v2.1 parquet/MP4 files.  This environment does not need
spline_encoder installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


def check(dataset: Path, frames: int) -> dict:
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    import lerobot

    info = json.loads((dataset / "meta/info.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo_id = "local/piper_check"
        shutil.copytree(dataset, root / repo_id)
        convert_dataset(repo_id=repo_id, root=root, push_to_hub=False, force_conversion=True)
        # pyav avoids depending on system FFmpeg libraries for torchcodec.
        loaded = LeRobotDataset(repo_id, root=root / repo_id, video_backend="pyav")
        assert loaded.num_episodes == info["total_episodes"], "episode count differs"
        assert loaded.num_frames == info["total_frames"], "frame count differs"
        video_keys = sorted(k for k, v in info["features"].items() if v["dtype"] == "video")
        assert sorted(loaded.meta.video_keys) == video_keys, "video keys differ"
        expected = []
        for path in sorted((dataset / "data").rglob("*.parquet")):
            expected.append(pq.read_table(path).to_pandas())
        import pandas as pd

        table = pd.concat(expected, ignore_index=True)
        hf = loaded.hf_dataset.with_format(None)
        for column in ("observation.state", "action"):
            assert np.array_equal(np.asarray(hf[column], np.float32),
                                  np.stack(table[column].to_numpy()).astype(np.float32)), column
        for column in ("frame_index", "episode_index", "index", "task_index"):
            assert np.array_equal(np.asarray(hf[column]).reshape(-1), table[column].to_numpy()), column
        checked = 0
        if video_keys:
            import cv2

            for index in np.linspace(0, len(loaded) - 1, frames).astype(int):
                item = loaded[int(index)]
                row = table.iloc[int(index)]
                for key in video_keys:
                    video = dataset / info["video_path"].format(
                        episode_chunk=int(row["episode_index"]) // info["chunks_size"],
                        video_key=key, episode_index=int(row["episode_index"]))
                    capture = cv2.VideoCapture(str(video))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))
                    ok, frame = capture.read()
                    capture.release()
                    assert ok, video
                    image = (item[key].numpy().transpose(1, 2, 0) * 255.0)[..., ::-1]
                    difference = np.abs(image - frame.astype(np.float64)).mean()
                    assert difference < 4.0, f"{key} frame {index}: mean abs diff {difference}"
                    checked += 1
        stats = loaded.meta.stats
        assert all(k in stats for k in ("observation.state", "action", *video_keys)), "stats"
        return {"passed": True, "lerobot_version": lerobot.__version__,
                "episodes": loaded.num_episodes, "frames": loaded.num_frames,
                "video_keys": video_keys, "frames_compared": checked,
                "reader": "lerobot convert_dataset_v21_to_v30 + LeRobotDataset"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(check(args.dataset.resolve(), args.frames), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
