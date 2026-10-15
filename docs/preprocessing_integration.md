# Piper preprocessing integration

| | |
|---|---|
| Base commit | `5aede68` plus the review changes (see `docs/implementation_review.md`) |
| Date | 2026-10-15 |
| Source scripts | originally `src/data/*.py`; now byte-identical in `scripts/legacy_piper/` (legacy reference) |
| Example raw data | `DATA/classify_blocks` (85 episodes, read-only, gitignored: not in the repository) |
| State | Committed and fast-forwarded into `main` on `origin`; no package published |

## 1. What was integrated

The supplied scripts (first in `src/data`, now in `scripts/legacy_piper/`) are:
- converters: `convert_piper_to_lerobot.py` and `convert_piper_to_lerobot_delta_ee.py`;
- idle cleanup: `clean_static_frames_LeRobot.py`;
- the unified entry point: `preprocess_piper.py`;
- validators: one generic validator plus three for fixed historical dataset families;
- tests: `test_preprocess_piper.py`.

Their logic now lives in the top-level `data` package (a sibling of
`spline_encoder`; see §10 for the reorganization), with no `sys.path`
manipulation:

| Module | Ported from | Content |
|---|---|---|
| `data/conversion/raw.py` | converters, `preprocess_piper.py` | `data.pkl` reading, timestamp validation, nearest-timestamp selection, episode discovery |
| `data/conversion/media.py` | converters | OpenCV/ffmpeg video metadata and resampled encoding (identical ffmpeg command), image statistics |
| `data/conversion/episode.py` | `preprocess_piper.py` | per-episode alignment, state/action construction, cropping |
| `data/conversion/export.py` | converters | LeRobot v2.1 parquet, features, `episodes_stats.jsonl`, provenance files |
| `data/transforms/spaces.py` | converters, `preprocess_piper.py` | quaternion canonicalization, joint/EE deltas, modality layout, units |
| `data/transforms/cleanup.py` | `clean_static_frames_LeRobot.py` | motion signals, sustained-motion cutoff, metrics, summary, motion-source policy |
| `data/validation/dataset.py` | `validate_piper_preprocessing.py` | independent reconstruction from raw data and full dataset checks |
| `data/validation/sidecar.py` | new | decode-and-compare verification of spline sidecars |
| `data/splines/sidecar.py` | new | sidecar via the existing `create_encoder`/`encode_lerobot_dataset` |
| `data/splines/lerobot.py`, `data/splines/cli.py` | `spline_encoder/lerobot.py`, `cli.py` (moved) | LeRobot action I/O, `encode_lerobot_dataset`, `spline-encode` |
| `data/config.py`, `data/pipeline.py`, `data/cli.py` | `preprocess_piper.py` | configuration, planning, staging, validation, atomic swap; CLI |

Entry points:
- console script `spline-preprocess-piper`;
- `python -m data`;
- optional extra `pip install spline-encoder[piper]` (OpenCV headless, pandas, PyArrow).

Importing `spline_encoder` does not import `data`. Inside `data`, OpenCV,
ffmpeg, PyArrow, and pandas are imported lazily, only when a run needs them.

Not ported as package code: the three dataset-family validators
(`validate_action_ee_datasets.py`, `validate_action_joint_datasets.py`,
`validate_state_abs_datasets.py`). They hard-code historical task names, a
30 Hz clock, and fixed directory families. The generic validator subsumes their
value checks for any configuration. The cleanup script's standalone
"re-clean an existing LeRobot dataset" mode and its comparison MP4 were also
not ported; the pipeline cleans raw episodes before export, as
`preprocess_piper.py` does.

## 2. Configuration

One JSON file describes a run (TOML on Python ≥ 3.11). Command-line flags
override file values and keep the names and defaults of `preprocess_piper.py`.
Relative paths in a config resolve from the working directory, so run the
example from the repository root.
See `examples/piper_preprocess.json`:

```json
{
  "raw_root": "DATA/classify_blocks",
  "output_root": "outputs/piper",
  "setups": ["state-delta-joint"],
  "cameras": {"global": "observation.images.global", "hand": "observation.images.hand"},
  "run": {"fps": 30.0, "cleanup": true, "cleanup_motion": "auto", "min_remaining_seconds": 2.0},
  "spline": {"enabled": true, "mode": "uniform_left", "chunk_size": 16, "degree": 3,
             "num_basis": 11, "span_length_steps": 2, "validation_episodes": 5},
  "jobs": 16
}
```

| Key | Meaning (default) |
|---|---|
| `raw_root` | A folder of task folders (`DATA` with `tasks`), or one task folder (`DATA/classify_blocks`) |
| `tasks`, `task_description` | Task subfolders. The instruction defaults to the historical `TASKS` table; a description is required for unknown tasks. |
| `output_root` / `output` | Preset family layout `<root>/<Family>/<task>_<fps>hz[_cleanup]` (unchanged from the scripts), or one explicit path |
| `setups` | Preset names (`--list-setups`). `setup_overrides` changes preset fields while keeping the family path, like `--setup X --action-offset 0` in the scripts. |
| `custom` | A free `Setup`, written to `Custom/<state>__<action>__offset..__quat..__last..__motion..` |
| `cameras` | List of names, `{name: key}`, or full `{name, file, key, timestamps_key}` (default `global`, `hand`; `[]` for none) |
| `extra_observations` | `{lerobot_key: raw_observation_field}`, e.g. `observation.joint_velocity: arm_joint_velocity` |
| `run` | `fps` 30; cleanup on; motion source `auto`; thresholds arm 0.005 rad, gripper 0.01, EE 0.0005 m / 0.005 rad; confirmation 0.4 s at fraction 0.5; preroll 0; scan 10 s; minimum retained 2 s |
| `spline` | `enabled`, `mode`, `preset` or `encoder_state`, geometry (CLI defaults of `spline-encode`), `quantize`, `normalization`, `validation_episodes`, `output` |
| `validate`, `check_videos`, `limit_episodes`, `jobs`, `ffmpeg`, `overwrite`, `dry_run` | Run control. `jobs` is the number of parallel episode writers. |

### Spaces

| `Setup` field | Values | Layout |
|---|---|---|
| `state` | `joint` (7), `ee` (8), `last-command` (7), `joint-last-command` (14), `ee-last-command` (15) | measured joints+gripper[t]; measured TCP xyz+quaternion xyzw+gripper[t]; previous command |
| `action` | `joint`, `command` (7), `abs-ee` (8), `delta-joint` (7), `delta-ee` (7) | see §3 |
| `action_offset` | 0 or 1 | absolute actions at `t` or `t+1`; deltas require 1 |
| `quaternion` | `raw`, `continuous` | raw feedback, or normalized XYZW with continuous sign and first `w ≥ 0` |
| `last_command_clock` | `raw`, `target` | previous **raw** robot sample (historical) or previous target-clock sample |
| `delta_ee_frame` | `tool`, `base` | see §3 |
| `angle_unit` / `position_unit` | `radian`/`degree`, `meter`/`millimeter` | conversion applied once at export; quaternions and gripper unchanged; cleanup thresholds stay in raw units |

`modality.json` records every block's `unit`, `frame`, `absolute`,
`quaternion_order`, `quaternion_policy`, and command-history `clock`. Measured
EE always comes from `observations.arm_pos/arm_quat`. The command/FK EE fields
in raw `actions` are never read, and the synthetic tests poison them to prove
it.

## 3. Timing semantics

1. **Clock.** The target clock spans the overlap of the robot stream and the
   selected cameras only, at `fps`: `target = start + k/fps`, with
   `count = floor((end-start)·fps + 1e-9) + 1`. Every selected stream is
   resampled by nearest timestamp; ties choose the earlier sample. Unselected
   cameras are never opened or required and do not shrink the overlap (tested
   with a missing video and a short-lived third camera).
2. **Observation time.** Resampling happens before any difference. Images,
   state, and extra observations are at `t`.
3. **Absolute actions** are the resampled trajectory at `t + action_offset`.
   Presets keep their historical offsets: 1 for the `state-*` families, 0 for
   the `ActionCommand/ActionJoint/Processed` families.
4. **Deltas** are `t → t+1` on the resampled measured streams:
   - joints: `dq = q[t+1] − q[t]` in radians;
   - EE, tool frame (default): `dp = R[t]ᵀ(p[t+1] − p[t])` and
     `dr = Log(R[t]ᵀ R[t+1])` as a rotation vector;
   - EE, base frame: `dp = p[t+1] − p[t]` and `dr = Log(R[t+1] R[t]ᵀ)`;
   - all delta actions end with the **absolute measured gripper[t+1]**, not a
     gripper difference.

   The final resampled row, which has no next target, is dropped.
5. **Quaternions.** With `continuous`, quaternions are canonicalized on the raw
   clock and again after resampling (historical behavior), so skipped or
   repeated rows cannot flip signs. `raw` exports feedback unchanged.
6. **Command history.** The first row repeats the first command, since no
   earlier command exists.

## 4. Leading-idle cleanup

- **Scope.** Cleanup runs per episode and removes only a leading prefix
  `[0, cutoff)`. Internal pauses are kept, and separated segments are never
  joined: the output is always the contiguous suffix `aligned[cutoff:]` (tested
  with a 1 s internal pause).
- **Motion source.** `auto` selects command joints for `command` actions,
  measured EE deltas for `delta-ee`, measured joint deltas for `delta-joint`,
  and measured joints otherwise. Any of the four sources can be forced with
  `cleanup_motion`.
- **Onset detection.** A row is moving when a threshold is exceeded. The cutoff
  is the first moving row in the first window of
  `round(confirmation_seconds·fps)` rows that contains at least
  `round(window·fraction)` moving rows, minus `preroll`. It is capped so that
  `min_remaining_seconds` remain, and only the first `max_scan_seconds` are
  scanned.
- **Onset row.** Joint-trajectory signals mark transition `i → i+1`, so the
  first changed row is kept. Delta signals already describe motion starting at
  row `i`, so that row is kept. In both cases the motion-onset row is retained.
- **Edge cases.** All-static episodes, episodes shorter than the confirmation
  window, and motion after the scan limit all give cutoff 0. Episodes with fewer
  than two shared target samples are rejected.
- **Consistent cropping.** State, action, extra observations, every camera's
  frame indices, alignment errors, raw robot indices, and raw-clock times are
  cropped by the same cutoff. Timestamps and frame/global indices restart at 0.
  `meta/cleanup_metrics.csv`, `meta/cleanup_summary.json`,
  `meta/source_episodes.jsonl` (raw path, resampled frames, dropped tail,
  removed prefix), and `meta/source_alignment/...` keep the removal metrics and
  the mapping back to raw data.

## 5. Outputs

```text
<output_root>/<Family>/<task>_30hz_cleanup/            LeRobot v2.1
  data/chunk-000/episode_000000.parquet                 observation.state, [extras], action,
                                                        timestamp, frame_index, episode_index,
                                                        index, task_index
  videos/chunk-000/<camera key>/episode_000000.mp4      h264 yuv420p at fps (selected cameras)
  meta/info.json            codebase_version v2.1, features, fps, splits, totals, layouts,
                            preprocessing config (version 2), cleanup summary
  meta/tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
  meta/modality.json, embodiment.json                   GR00T/starVLA metadata
  meta/cleanup_summary.json, cleanup_metrics.csv, source_episodes.jsonl, validation.json
  meta/source_alignment/chunk-000/episode_000000.parquet
<output_root>/<Family>/<task>_30hz_cleanup_spline_<mode>/   optional sidecar
  encoder.json, manifest.json, data/episode_000000.npz      unchanged encode_lerobot_dataset schema
  preprocessing_provenance.json                             additive
```

`episodes_stats.jsonl` follows LeRobot v2.1:
- numeric features: per-episode min/max/mean/std/count;
- video features: per-channel RGB statistics in `[0, 1]` with shape `(3,1,1)`,
  from LeRobot's sampling and downsampling rule.

Provenance never enters `data/`, so LeRobot readers see exactly the declared
features.

**Sidecars.**
- **Input.** `create_encoder` plus `encode_lerobot_dataset` run on the
  **final, cleaned** `action` column of the staged dataset. `action_dim` and
  `frequency_hz` come from the dataset; a preset, saved encoder, or setting
  that does not match is rejected before any work (tested).
- **No double transforms.** The pipeline never differences, shifts, or
  normalizes actions again. The only optional transform is one `q01_q99`
  normalization, recorded in the manifest's `alignment`.
- **Training data only.** Quantization calibration and normalization bounds use
  training episodes only. `validation_episodes` takes the last `k` episodes out
  of training and declares them as a `validation` split in `info.json`.
- **Contents.**
  - Uniform sidecars store per-record controls; the fixed geometry is in
    `encoder.json`.
  - Adaptive sidecars additionally store knots, `duration_steps`, and
    `control_duration_interleaved`.
  - All sidecars keep `episode_index`, `frame_index`, and `observation_index`.
- **Provenance and tail policy.**
  - The manifest's `source_dataset` points to the final dataset, not the
    staging copy.
  - `preprocessing_provenance.json` records the dataset fingerprint, raw task,
    setup, tail policy, training episodes, calibration source, and decoding
    verification.
  - Tail policy: uniform windows repeat the final action; the adaptive refit
    pads with repeated final actions.
- **Quaternions.** Spline fitting is per-scalar, so decoded `abs-ee`
  quaternions are not unit length. The verification reports the norm deviation
  and says to renormalize.

**Staging and safety.**
- **Staging and swap.** Datasets and sidecars are written to hidden staging
  directories beside their destinations and validated. They are then swapped
  in together, with backup-and-restore on failure. If conversion, validation,
  or sidecar encoding fails, completed outputs stay byte-identical and staging
  is removed (tested with an injected sidecar failure).
- **Path checks.**
  - Outputs that overlap the raw tree, each other, or a sidecar are rejected.
  - Existing outputs require `--overwrite`.
  - Raw inputs are only opened for reading.

## 6. Commands and results

Environment:

| | |
|---|---|
| Python env | conda `starVLA`: Python 3.10.19, NumPy 1.26.4, SciPy 1.15.3, OpenCV 4.11.0, pandas 2.3.3, PyArrow 14.0.1 |
| ffmpeg | imageio-ffmpeg 7.0.2 binary from the conda base env (found automatically) |
| CPU | 384 cores |

`PYTHONPATH=src` is used because this env's editable install points at another
checkout.

### Real data: `DATA/classify_blocks` (85 episodes, 2.2 GB, read-only)

Each episode holds `data.pkl` (robot 100 Hz, schema v3 with per-camera
timestamps) plus `global_image.mp4` and `hand_image.mp4` (640×480, 30 Hz).

| Run | Command | Wall time | Result |
|---|---|---|---|
| A. example config | `spline-preprocess-piper --config examples/piper_preprocess.json` | 1 min 50 s (16 writers) | `StateDeltaJoint`: 85 episodes, 190,404 aligned rows, 4,406 idle rows removed (2.3%; every episode clipped; median 1.63 s, range 0.67–3.2 s), 185,998 frames, 170 videos, 860 MB; split train 0:80 / validation 80:85; uniform-left sidecar 86 MB |
| B. preset + adaptive | `--raw-root DATA/classify_blocks --setup state-abs-ee --spline --spline-mode adaptive_left --spline-quantize --spline-validation-episodes 5` | 8 min 9 s | `StateAbsEE`: 185,913 frames, 170 videos; adaptive quantized sidecar |
| C. continuous quaternions | `--state joint --action abs-ee --quaternion continuous` + same spline flags | 6 min 21 s | `Custom/joint__abs-ee__offset1__quat-continuous__…`: 185,913 frames |
| D. dry run | `--config examples/piper_preprocess.json --dry-run` | < 1 s | plans 85 episodes / 185,998 frames / 4,406 removed; writes nothing |

The built-in independent validator passed for A–C, checking every row against
the raw pickles and all 170 videos each.

Sidecar decoding against the cleaned actions (from
`preprocessing_provenance.json`; all records decode to finite values):

| Sidecar | Records | Fitting RMSE / p95 / max | Dequantized RMSE / max | Controls outside training calibration (train / held-out) | Quaternion |
|---|---|---|---|---|---|
| A uniform-left, delta-joint | 185,998 | 2.07e-3 / 3.4e-3 / 0.095 (gripper; arm ≤ 0.027) | — | — | — |
| B adaptive-left, abs-ee, `raw` quaternions | 185,913 | 7.3e-4 / 1.5e-3 / 9.9e-3 | 2.2e-3 / 0.48 | 0 / 981 | 2,520 sign flips; decoded norm deviation ≤ 0.0102; warning emitted |
| C adaptive-left, abs-ee, `continuous` | 185,913 | 7.8e-4 / — / 9.9e-3 | 1.5e-3 / 0.22 | 0 / 1,815 | no flips; norm deviation ≤ 0.0011; median knot duration 5 steps (B: 2) |

Independent cross-checks on the final artifacts:

| Check | A | B | C |
|---|---|---|---|
| `scripts/validate_lerobot_encoding.py` full recomputation | exact, 185,998 obs | exact, 185,913 obs | exact, 185,913 obs |
| `scripts/check_lerobot_reader.py` (LeRobot 0.4.5 convert + load, `pyav`) | pass: 85 episodes, 185,998 frames, rows equal, 10 video frames | pass: 85 episodes, 185,913 frames | — |

**Finding.** The historical `state-abs-ee` preset exports **raw** feedback
quaternions, and Piper's TCP feedback flips sign within every episode
(qy ≈ +0.98 ↔ −0.98; 2,520 flips in the dataset). The LeRobot dataset is
faithful to the preset, but scalar splines then need many short knots, and
quantization bounds become very wide. The preset is unchanged for
compatibility. The sidecar verification counts the flips and recommends
`--quaternion continuous` (run C). Large dequantized maxima in B and C come
from clipping on the five held-out episodes, which the training-only
calibration never saw; training-episode clipping is zero.

### Legacy equivalence on real data

```bash
python -B scripts/legacy_piper/preprocess_piper.py --raw-root DATA --tasks classify_blocks \
  --setup state-delta-joint --setup state-delta-ee --setup joint-last-command-command \
  --limit-episodes 2 --output-root LEGACY --jobs 3
spline-preprocess-piper --raw-root DATA --tasks classify_blocks \
  --setup state-delta-joint --setup state-delta-ee --setup joint-last-command-command \
  --limit-episodes 2 --output-root NEW --jobs 4
python scripts/compare_piper_preprocessing.py LEGACY/<family>/classify_blocks_30hz_cleanup NEW/<family>/...
```

All three presets passed: 5,003 / 5,005 / 5,012 rows, and 4/4 MP4 files
byte-identical per preset. Both pipelines' validators passed. Running
`check_lerobot_reader.py` on the **legacy** output fails with
`FileNotFoundError: meta/episodes_stats.jsonl`.


### Tests

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider -rs
```

Final run: **382 passed, 2 skipped** in 14.4 s. Before this work the suite
was 174 tests: 172 passed and 2 skipped.

| File | Tests | Coverage |
|---|---:|---|
| `tests/test_piper_preprocessing.py` | 200 | Ported semantics: all 160 state/action/offset/quaternion/clock combinations against the independent reconstruction, cleanup sources, command clocks, presets, idle cutoff independent of state, measured-only recordings, delta-cleanup onset. New: delta-EE reconstruction in tool and base frames, delta-joint reconstruction, quaternion policies, joint-cleanup onset row, all-static/short/too-short episodes, internal pause kept with a contiguous crop, unselected cameras neither loaded nor bounding overlap (missing file, short third camera), extra observations, end-to-end v2.1 export for 4 presets with `episodes_stats`, camera subsets (none / renamed hand / global+wrist) with video counts, cleanup and provenance alignment against raw indices, unit conversion, overwrite refusal, failure preserving completed dataset+sidecar, overlapping/duplicate paths, task-folder roots, dry run and limits, config file and CLI overrides |
| `tests/test_piper_splines.py` | 9 | uniform-left / uniform-double / adaptive-left sidecars re-encoded independently and compared with stored controls/knots and episode/frame mapping; decoding error and finiteness; LeRobot actions unchanged by the sidecar; training-only normalization and quantization; quaternion norm and sign-flip reporting; held-out split reporting; incompatible encoder state, preset, and settings rejected before writing; compatible saved encoder reused |
| `tests/test_piper_legacy_compat.py` | 1 | bit-equality with the unmodified `scripts/legacy_piper` functions (subprocess) for 160 combinations + 12 presets + both legacy converters |
| existing 14 files | 174 | unchanged: 172 passed, 2 skipped |

The `-rs` skip reasons are the two mathematically inapplicable parameter
cases from `test_reference_geometry.py`. The 123 warnings are pandas
`DeprecationWarning`s from `pd.read_parquet`.


## 7. Compatibility

- **Encoder untouched.** No encoder API, signature, default, numerical
  convention, `spline-encode` flag, or serialized format changed.
  `spline_encoder.__all__` is unchanged. The sidecar uses the existing public
  functions and schema; `preprocessing_provenance.json` is a new extra file.
- **Legacy equivalence.**
  - `tests/test_piper_legacy_compat.py` runs the unmodified `scripts/legacy_piper`
    functions in a subprocess. Per-episode state, action, cutoff, and camera
    frame selection are bit-identical for all 160 state/action/offset/
    quaternion/clock combinations (5 states × 8 action/offset pairs × 2
    quaternion policies × 2 clocks, cycling through four cleanup
    configurations) and all 12 presets. The legacy delta-EE and joint
    converters also match.
  - On real data, `scripts/compare_piper_preprocessing.py` found bit-identical
    outputs against legacy `preprocess_piper.py` (3 presets × 2 episodes; see
    §6), including byte-identical MP4 files.
- **Intentional dataset differences from the legacy outputs** (all additive or
  corrective):
  - `meta/episodes_stats.jsonl` is new. The legacy datasets cannot be loaded by
    LeRobot's v2.1 reader without it (`FileNotFoundError` reproduced with
    LeRobot 0.4.5).
  - `features.*.names` lists one name per dimension instead of
    `["state"]`/`["action"]`.
  - `modality.json` gains descriptive keys: `quaternion_policy`, `frame` on the
    EE quaternion, and `clock` on command history.
  - `preprocessing.version` is 2, with `package`, `cameras`, and
    `extra_observations`.
  - New `meta/source_alignment/*` files.
  - `jobs` parallelizes episode writing within a dataset instead of running
    several datasets concurrently.
  - `total_chunks` and chunk paths are correct beyond 1000 episodes; the
    scripts hard-coded `chunk-000`.
- **Optional dependencies.** `import spline_encoder` works without OpenCV,
  pandas, PyArrow, or ffmpeg. A camera-less pipeline needs no OpenCV or ffmpeg.

## 8. Limitations

- **Uniform clock.** The pipeline resamples onto a uniform clock with
  nearest-sample selection (no interpolation). Alignment error is bounded by
  half the source period plus timestamp jitter and is recorded per row.
- **Dataset scope.** The exporter writes one task per dataset and
  `chunks_size` 1000. Images are always re-encoded as h264/yuv420p (lossy) at
  the target FPS.
- **Image statistics.** Computed from sampled, downsampled frames as LeRobot
  does. They are approximate, not exact full-video statistics.
- **Gripper.** Gripper values are passed through as recorded (normalized
  `[0, 1]` in this data). Cleanup thresholds are in raw radians/meters even
  when export units differ.
- **Sidecar error bounds.** Fitting errors are measured, not bounded. Steps in
  the gripper or delta signals produce the largest errors (see §6); adaptive
  splines follow steps better than uniform ones. Quaternion components are
  fitted independently, as noted.
- **Quantization.** Uses the existing per-row (uniform) or per-dimension
  (adaptive) min/max quantizer; there is no percentile calibration.
- **Validation split.** Defined by episode order (the last `k`), not by random
  or task-aware splitting.
- **LeRobot reader check.** Requires an environment with LeRobot ≥ 0.4 and a
  NumPy-compatible SciPy. The local `fastumi` env has SciPy 1.10.1 with NumPy
  2.2.6, so the check ran in an isolated scratch venv inheriting that env plus
  SciPy 1.15.3; no user environment was modified. The check converts a temporary
  copy to v3.0 with LeRobot's own converter and loads it; it does not exercise a
  native v2.1 LeRobot (< 0.4) installation.
- **starVLA loader.** starVLA's GR00T-style loader was not executed. Only its
  metadata requirements were inspected: `modality.json`, `info.json` video
  `info`, and statistics computed into `stats_gr00t.json`.
- **Raw quaternions in historical presets.** `state-abs-ee` and other
  `quaternion="raw"` setups export sign-flipping feedback quaternions (see §6).
  This is kept for compatibility, flagged by the sidecar verification, and
  avoided with `--quaternion continuous`.
- **Leftover scripts.** `scripts/legacy_piper` still holds the original scripts. Deleting
  them, or replacing them with thin wrappers around the package, is left to
  the maintainers.

## 9. Changed files

New, packaged:
- `src/data/` (see §10; originally written as `src/spline_encoder/piper/`: `__init__`, `__main__`, `cli`, `config`, `raw`,
  `cleanup`, `episode`, `media`, `export`, `validate`, `splines`, `pipeline`).

New scripts and examples:
- `scripts/compare_piper_preprocessing.py`
- `scripts/check_lerobot_reader.py`
- `examples/piper_preprocess.json`

New tests:
- `tests/piper_fixtures.py`
- `tests/test_piper_preprocessing.py`
- `tests/test_piper_splines.py`
- `tests/test_piper_legacy_compat.py`

Modified:
- `pyproject.toml`: `piper` extra, OpenCV in the `test` extra, the
  `spline-preprocess-piper` console script, and (§10) both packages in the
  wheel.
- `.gitignore`: `outputs/` and `DATA/`.
- `README.md`: Piper section in English and Chinese, repository layout.

Docs:
- this file.

Not modified:
- `scripts/legacy_piper/*` (legacy reference, moved byte-identically from `src/data/`) and `DATA/` (raw input; gitignored, never committed).

Reorganization (§10):
- encoder modules moved to `spline_encoder/{core,encoding,tools}/`; only
  import lines changed, plus a function-local `SplineParameters` import in
  `core/subdivision.py`;
- `spline_encoder/__init__.py` gained the alias table and lazy LeRobot names;
- `spline_encoder/{lerobot,cli}.py` are now alias modules;
- `pyproject.toml` package list and entry points updated;
- test imports updated;
- `README.md` and `docs/implementation_review.md` updated.

The earlier review's numerical fixes are documented in
`docs/implementation_review.md`.

Gitignored, not part of the change: the raw recordings in `DATA/` and the generated `outputs/piper/...` from runs
A–C (about 2.6 GB).


## 10. Package reorganization (follow-up)

The repository was reorganized into two sibling top-level packages under
`src/`. This was a code-organization change only: no feature was added and no
numerical code was edited (only module locations and import lines changed).

```text
src/
├── spline_encoder/            B-spline representation (numpy/scipy; matplotlib optional)
│   ├── __init__.py            public API, unchanged 59-name __all__; aliases for old module paths;
│   │                          LeRobot names re-exported lazily from data.splines.lerobot
│   ├── __main__.py            python -m spline_encoder  (== spline-encode)
│   ├── cli.py, lerobot.py     compatibility aliases -> data.splines.cli / data.splines.lerobot
│   ├── core/                  adaptive.py, fitting.py, subdivision.py
│   ├── encoding/              config.py, encoder.py, result.py, quantization.py, serialization.py
│   └── tools/                 verification.py, visualization.py
└── data/                      robot data processing (may import spline_encoder)
    ├── __init__.py, __main__.py, cli.py, config.py, pipeline.py   spline-preprocess-piper / python -m data
    ├── conversion/            raw.py, media.py, episode.py, export.py
    ├── transforms/            spaces.py, cleanup.py
    ├── validation/            dataset.py, sidecar.py
    └── splines/               lerobot.py, sidecar.py, cli.py
scripts/legacy_piper/          the 9 supplied scripts, byte-identical (sha256 verified), not packaged
```

**Responsibilities and dependency rule.**
- `spline_encoder.core`: knots, least-squares and FITPACK fitting, knot
  insertion and subdivision. It has no import-time dependency on `encoding`;
  `subdivide_parameters` imports `SplineParameters` inside the function.
- `spline_encoder.encoding`: configurations, encoders, results, quantization,
  and serialization.
- `spline_encoder.tools`: verification reports and plots.
- None of these three groups imports `data`, OpenCV, pandas, or PyArrow (checked
  by grep and by `sys.modules` after `import spline_encoder`).
- `data` imports `spline_encoder.encoding` modules directly, never via the
  top-level package, which avoids import cycles.
- The only `spline_encoder` → `data` references are the lazy compatibility
  aliases (`cli.py`, `lerobot.py`, and `__getattr__` for 9 LeRobot names).
- Group `__init__` files contain docstrings only, so no group re-imports its
  siblings.
- A future policy-inference package would be another sibling under `src/`; no
  placeholder was created.

**Old → new mapping.**

| Old location | New location | Compatibility |
|---|---|---|
| `spline_encoder/{fitting,adaptive,subdivision}.py` | `spline_encoder/core/` | old import paths are the same module objects (alias table in `__init__`) |
| `spline_encoder/{config,encoder,result,quantization,serialization}.py` | `spline_encoder/encoding/` | same |
| `spline_encoder/{verification,visualization}.py` | `spline_encoder/tools/` | same |
| `spline_encoder/lerobot.py` | `data/splines/lerobot.py` | `spline_encoder.lerobot` alias + lazy top-level names |
| `spline_encoder/cli.py` | `data/splines/cli.py` | `spline_encoder.cli` alias; `spline-encode` entry point now `data.splines.cli:main` |
| `spline_encoder/piper/{config,pipeline,cli,__main__}.py` | `data/` | not shimmed (never committed) |
| `spline_encoder/piper/{raw,media,episode,export}.py` | `data/conversion/` | quaternion/delta functions → `data/transforms/spaces.py`; layout/units/modality → `spaces.py`; `motion_source` → `transforms/cleanup.py` |
| `spline_encoder/piper/cleanup.py` | `data/transforms/cleanup.py` | — |
| `spline_encoder/piper/validate.py` | `data/validation/dataset.py` | — |
| `spline_encoder/piper/splines.py` | `data/splines/sidecar.py` (+ `verify_sidecar` → `data/validation/sidecar.py`, re-exported) | — |
| `src/data/*.py` (supplied scripts) | `scripts/legacy_piper/*.py` | run as scripts; `src/data` is now the package |

Packaging: `[tool.hatch.build.targets.wheel] packages = ["src/spline_encoder",
"src/data"]`. Console scripts are `spline-encode = data.splines.cli:main` and
`spline-preprocess-piper = data.cli:main`. Extras are unchanged.

**Validation of the reorganization.**

| Check | Result |
|---|---|
| Test suite, source tree (`PYTHONPATH=src`, starVLA: NumPy 1.26.4, SciPy 1.15.3) | 382 passed, 2 skipped (identical to before) |
| Every module imported alone in a fresh interpreter (35 modules) | all import; no cycles |
| Old paths `spline_encoder.{fitting,…,visualization,lerobot,cli}` | identical module objects; private names (`_design`, `_boehm_insert`, `_custom_config`, …) reachable; all 59 `__all__` names resolve |
| `import spline_encoder` loads `data`/cv2/pandas/pyarrow/matplotlib? | no |
| Encoder output snapshot (`compare_output_snapshots.py`, 546 arrays, 807,784 values) | bit-identical to the pre-reorganization tree and to `5aede68` |
| Pipeline outputs, 2 real episodes, 3 presets, both spline modes, vs pre-reorganization code | parquet 20/20, MP4 20/20, NPZ 4/4, JSONL, CSV, `encoder.json`, and manifests identical. Only intended difference: `info.json` `preprocessing.package` is `"data"` (was `"spline_encoder.piper"`), so the sidecar's `dataset_fingerprint` (which hashes `info.json`) changes too. |
| Same outputs vs the legacy scripts (`compare_piper_preprocessing.py`) | passed; videos byte-identical |
| `spline-encode --help` | byte-identical before/after and from the installed wheel |
| Wheel (`pip wheel .`) | 37 modules (`spline_encoder/**`, `data/**`) + metadata; no tests, scripts, legacy files, data, or bytecode |
| Clean venv from the wheel `[piper,visualization,test]` (NumPy 2.2.6, OpenCV 5.0.0, PyArrow 25.0.1), run outside the checkout | imports resolve to site-packages; `spline-encode`, `python -m spline_encoder`, `spline-preprocess-piper`, and `python -m data` all work; repository tests against the installed wheel: 383 passed (382 + origin check), 2 skipped |
| Installed CLI on 2 real episodes vs source run on starVLA | parquet tables, NPZ arrays, and MP4 bytes identical. One cleanup-diagnostic CSV cell (a float32 p95 quantile) differs by 9.3e-11 between NumPy 1.26 and 2.2; parquet file bytes differ only in the PyArrow writer stamp (14 vs 25), which also changes the sidecar `dataset_fingerprint`. |

**Caveats.**
- `data` is a generic top-level name. A `data/` directory or `data.py` earlier on
  `sys.path`, for example the current working directory, shadows the installed
  package; starVLA's repository root has such a directory. Run the CLIs from
  elsewhere or rename the package later. The requested layout was kept.
- Artifacts produced before this reorganization carry
  `"package": "spline_encoder.piper"` in `info.json`.
