# BSplineEncoder

BSplineEncoder is an independent, framework-neutral Python package that turns
robot action chunks into observation-aligned B-spline parameters. Continuous
control points and knots are the primary representation; 256-bin (or custom)
quantization is an optional final step.

It accepts local standard LeRobot v2.1 and v3.0 datasets without importing the
LeRobot training stack. The numerical implementations preserve the validated
dRTCv2 tokenizer contracts, including the Piper 50 Hz presets and legacy
`action_tokenizer.json` files.

## Install

```bash
git clone https://github.com/outsider86/BSplineEncoder.git
cd BSplineEncoder
pip install -e ".[lerobot]"
```

The core action-chunk API needs only NumPy and SciPy. Install the `lerobot`
extra to read parquet datasets, and `test` to run the full suite.

## Encode one action chunk

```python
import numpy as np
from spline_encoder import encode_action_chunk, preset

# Shape (56, 7): uniform-left uses 50 executable actions plus fitting context.
actions = np.load("action_window.npy")
config = preset("piper_chunk50", mode="uniform_left")
result = encode_action_chunk(actions, config)

assert result.control_points.shape == (28, 7)
assert result.knots.shape == (32,)
assert result.decode().shape == (50, 7)
```

`result.control_points` and `result.knots` are lossless continuous model
parameters. Knots are in seconds; `result.knot_steps` gives action timesteps.

For repeated calls, construct one encoder so the solver is reused:

```python
from spline_encoder import create_encoder

encoder = create_encoder(config)
result = encoder.encode_chunk(actions)  # continuous, no calibration needed
```

## B-spline options

Three geometries are public:

| Mode | Input per observation | Stored parameters | Executable output |
|---|---:|---:|---:|
| `uniform_double` | `chunk_size` | fixed knots, `num_basis` controls | `chunk_size` |
| `uniform_left` | longer fitting window | fixed left-clamped knot prefix and controls | `chunk_size` |
| `adaptive_left` | episode fit, or one standalone chunk | adaptive durations and controls | duration of first `num_basis - degree` spans |

All configurations expose degree, action dimension, frequency, number of
basis functions, vocabulary size, and fitting controls. For example:

```python
from spline_encoder import UniformDoubleBSplineConfig, create_encoder

config = UniformDoubleBSplineConfig(
    action_dim=12,
    chunk_size=24,
    frequency_hz=30,
    degree=3,
    num_basis=11,
    span_length_steps=3,  # (11 - 3) * 3 == 24
    regularization=1e-5,
)
encoder = create_encoder(config)
```

Ready-to-use presets are `libero`, `libero_chunk8`, `robotwin`, and
`piper_chunk50`.

### Why uniform-left consumes more actions

A cubic local spline needs three controls of right-side support. The Piper
uniform-left preset fits a 56-action, 31-control double-clamped spline, then
stores controls 0–27. Those 28 controls completely determine actions 0–49;
the three fitting-only suffix controls are discarded. The representation is
therefore observation-aligned to a 50-action execution chunk even though the
fit consumes 56 source actions.

## Optional quantization

Calibration and quantization are explicit:

```python
windows = [...]  # iterable of correctly sized aligned action windows
encoder.calibrate(windows)
encoded = encoder.encode_chunk(actions, quantize=True)

encoded.tokens                  # integer (num_basis, action_dim)
encoded.control_points           # original continuous fit
encoded.dequantized_control_points
encoded.decode(dequantized=True) # reconstruction from tokens
```

Uniform encoders calibrate each control row and action dimension independently.
Adaptive encoders calibrate per action dimension, matching the dRTCv2
tokenizer. Saved legacy dRTCv2 artifacts can be loaded directly:

```python
from spline_encoder import load_encoder

encoder = load_encoder("action_tokenizer.json")
result = encoder.encode_chunk(actions, quantize=True)
```

## Encode a LeRobot dataset

Dimension selection/reordering and normalization are represented by an
`ActionAlignment`. Every source observation receives the future action chunk
starting at that observation; episode ends are padded by repeating the final
action.

```python
from spline_encoder import (
    ActionAlignment,
    create_encoder,
    encode_lerobot_dataset,
    preset,
)

alignment = ActionAlignment(
    action_key="action",
    action_indices=(0, 1, 2, 3, 4, 5, 6),
)
encoder = create_encoder(preset("piper_chunk50", "uniform_left"))

summary = encode_lerobot_dataset(
    "/datasets/piper_pick",
    "encoded_splines/piper_pick",
    encoder,
    alignment=alignment,
)
print(summary.record_count)
```

To reproduce a robust `[-1, 1]` Piper alignment from the 1st/99th percentiles:

```python
from spline_encoder import infer_action_bounds

alignment = infer_action_bounds(
    "/datasets/piper_pick",
    action_indices=(0, 1, 2, 3, 4, 5, 6),
)
```

Quantized dataset export performs a calibration pass when requested:

```python
encode_lerobot_dataset(
    dataset,
    output,
    encoder,
    alignment=alignment,
    quantize=True,
    calibrate_quantizer=True,
)
```

The source dataset is never modified. Output contains:

```text
output/
├── encoder.json
├── manifest.json
└── data/
    ├── episode_000000.npz
    └── ...
```

Each NPZ has one row per source observation:

- `frame_index`, `observation_index`
- `controls[observation, basis, action_dimension]`
- `knots[observation, basis + degree + 1]`
- `executable_steps[observation]`
- `duration_steps` for left/adaptive modes
- `tokens` and `dequantized_controls` only when quantization is enabled

### CLI

```bash
spline-encode /datasets/piper_pick encoded_splines/piper_pick \
  --preset piper_chunk50 \
  --mode uniform_left \
  --action-indices 0,1,2,3,4,5,6 \
  --normalization q01_q99
```

Use `--quantize --calibrate` to add tokens, or `--encoder-state` to apply an
existing calibrated SplineEncoder/legacy tokenizer JSON.

## Exact subdivision

Low-level functions operate on SciPy splines and use exact Boehm knot
insertion:

```python
from spline_encoder import subdivide

left, right = subdivide(result.scipy_spline(), value=0.5)
```

For a double-clamped `SplineParameters`, split by native action step:

```python
from spline_encoder import subdivide_parameters

left, right = subdivide_parameters(result, step=25)
```

Evaluating the original and either subdivided curve on its domain agrees to
floating-point precision. `extract_interval`, `insert_knot`, and
`extract_left_clamped_prefix` are also public for real-time or causal systems.

## Verification

Run unit/integration tests:

```bash
pytest
```

The repository also includes an acceptance comparison against the original
dRTCv2 tokenizer and its complete 34,254-observation Piper caches:

```bash
python scripts/verify_original_tokenizer.py \
  --drtcv2-root /scratch/wangpc/dRTCv2 \
  --dataset /scratch/wangpc/dRTCv2/data/20260818_piper_pick_white_block_50hz
```

The report checks configuration fingerprints, continuous control points,
knots/durations, discrete tokens, observation keys, and full-cache coverage.
