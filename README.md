# BSplineEncoder

[English](#english) · [中文](#中文)

---

# English

## Overview

BSplineEncoder is a framework-neutral Python package for representing
multi-dimensional robot action sequences with B-splines. It contains:

- standalone control-point fitting from actions, timestamps, and a supplied
  knot vector;
- fixed-knot and adaptive encoders for observation-aligned action chunks;
- optional calibrated control-point quantization;
- local LeRobot v2.1/v3.0 dataset export;
- knot insertion, subdivision, verification, and visualization utilities;
- an explicit legacy loader for the dRTCv2 Piper uniform-left artifacts
  validated in this workspace.

The package is currently `0.1.0` and **Alpha**. Its API is usable but not
frozen. This is research/preprocessing code, not a safety-certified controller.

> **Honest scope:** control points and knots preserve the fitted spline, not
> necessarily the original sampled actions. Fitting normally introduces
> approximation error, and quantization adds more error. Measure reconstruction
> quality on your own data before using the representation in a policy.

## Installation

Core requirements are Python `>=3.10`, NumPy `>=1.23`, and SciPy `>=1.15`.

```bash
pip install "spline-encoder @ git+https://github.com/outsider86/BSplineEncoder.git"
```

For development, clone the standalone repository and install it in editable
mode. Optional extras keep dataset and plotting dependencies out of the core
installation:

```bash
git clone https://github.com/outsider86/BSplineEncoder.git
cd BSplineEncoder
pip install -e .
pip install -e ".[lerobot]"       # pandas and PyArrow
pip install -e ".[visualization]" # Matplotlib
pip install -e ".[test]"          # complete package test dependencies
```

This workspace has exercised the package in the `starVLA` Conda environment
with Python 3.10. The repository does not currently provide a complete CI
matrix for every advertised Python/NumPy/SciPy/platform combination.

## Mathematical conventions

For degree `k`, full knot vector `t`, and `n` control points `C`, this package
uses SciPy's convention:

```text
len(t) = n + k + 1
S(x)   = B(x; t, k) @ C
domain = [t[k], t[n]]
```

Typical shapes are:

```text
timestamps     (num_samples,)
actions        (num_samples, action_dim)
design matrix  (num_samples, num_basis)
control points (num_basis, action_dim)
```

All action dimensions share the same timestamps and knot geometry. They are
solved together in one vectorized least-squares operation. “Parallel” here
means a multi-output matrix solve, not one process/thread per dimension.

## Fit control points on your own knots

Use this API when you already know the timestamps and desired knot layout.

```python
import numpy as np

from spline_encoder import (
    evaluate_bspline,
    fit_control_points,
    make_clamped_knots,
)

timestamps = np.linspace(0.0, 2.0, 41)
actions = np.stack(
    (np.sin(timestamps), np.cos(timestamps), 0.25 * timestamps),
    axis=1,
)

# Input here is a list of distinct span boundaries. The helper repeats the
# endpoints to produce the full cubic knot vector.
knots = make_clamped_knots([0.0, 0.5, 1.0, 1.5, 2.0], degree=3)

controls = fit_control_points(actions, timestamps, knots, degree=3)
reconstructed = evaluate_bspline(controls, knots, timestamps, degree=3)

assert controls.shape == (7, 3)
assert reconstructed.shape == actions.shape
```

`make_clamped_knots()` accepts distinct, strictly increasing breakpoints.
`fit_control_points()` accepts the expanded full knot vector. The number of
controls is always `len(knots) - degree - 1`.

Use `fit_bspline()` for diagnostics and repeated evaluation:

```python
from spline_encoder import fit_bspline

fit = fit_bspline(
    actions,
    timestamps,
    knots,
    degree=3,
    weights=None,          # optional positive weight per timestamp
    regularization=1e-6,  # optional L2 control penalty
)

print(fit.control_points)
print(fit.design_rank, fit.condition_number)
print(fit.rmse, fit.max_absolute_error)

query_actions = fit(np.linspace(*fit.domain, 200))
query_times, query_actions = fit.sample(200)
scipy_spline = fit.scipy_spline()
```

The solved objective is:

```text
min_C ||W (B C - A)||² + λ ||C||²
```

The implementation uses augmented least squares rather than explicitly
forming normal equations. Inputs must be finite; fitting timestamps must be
strictly increasing and inside the spline domain; weights must be positive.
Degree is restricted to integers 1–5.

By default, `require_full_rank=True` rejects ambiguous control-point fits. Set
it to `False` only when a minimum-norm or regularized underdetermined solution
is intentional. A full-rank matrix can still be ill-conditioned; inspect
`condition_number`. Regularization can improve stability but biases controls,
so exact recovery should not be expected when it is nonzero.

Related public functions:

- `make_uniform_clamped_knots()`
- `make_uniform_left_clamped_knots()`
- `bspline_design_matrix()`
- `evaluate_bspline()`
- `fit_bspline_control_points` (alias of `fit_control_points`)

See `examples/fit_control_points.py` for a runnable example.

## Observation-aligned encoder API

```python
import numpy as np
from spline_encoder import create_encoder, preset

config = preset("piper_chunk50", mode="uniform_left")
encoder = create_encoder(config)

# Direct uniform-left fits 50 executable actions from the same 50-action chunk.
actions = np.load("action_window.npy")
assert actions.shape == (config.input_chunk_size, config.action_dim)

result = encoder.encode_chunk(actions)
assert result.control_points.shape == (28, 7)
assert result.knots.shape == (32,)
assert result.decode().shape == (50, 7)
```

`encode_action_chunk(actions, config)` is a one-call convenience function.
For repeated uniform fits, reuse one encoder because its solver is cached.

### Modes

| Mode | Required input | Stored representation | Native output |
|---|---|---|---|
| `uniform_double` | exactly `chunk_size` actions | fixed double-clamped knots and `num_basis` controls | exactly `chunk_size` samples |
| `uniform_left` | exactly `chunk_size` actions | fixed left-clamped/right-open knots and `num_basis` controls | exactly `chunk_size` samples |
| `adaptive_left` | complete episode, or `chunk_size` for standalone use | per-record controls and integer knot durations | variable horizon |

`uniform` is a legacy alias for `uniform_double`. `adaptive` uses the same
implementation as `adaptive_left`; explicit canonical names are clearer.

### Uniform double

With `span_length_steps` set, geometry is endpoint-exclusive:

```text
native samples: 0 ... chunk_size - 1
spline domain:  [0, chunk_size]
constraint:     (num_basis - degree) * span_length_steps == chunk_size
```

With `end_padding=True`, the final action is repeated at the right endpoint
during fitting. With `span_length_steps=None`, the older layout instead uses an
inclusive domain from sample `0` to `chunk_size - 1`.

The default `regularization=1e-4` means the encoder is intentionally a
regularized approximator, not an exact interpolator.

### Uniform left

`uniform_left` constructs its left-clamped/right-open knot vector directly and
fits all `num_basis` controls from the action chunk itself. For span length `s`,
degree `k`, and `n = num_basis`, the full knot vector in action-step units is:

```text
[0 repeated k times] + [0, s, 2s, ..., n*s]
```

Therefore zero has multiplicity `k + 1`, the SciPy base interval is
`[0, (n-k)*s] = [0, chunk_size]`, and the last `k` knots provide right-side
basis support without clamping the right boundary. The implementation builds
this matrix once and solves directly for all controls. There is no extended
action window, double-clamped intermediate spline, retained prefix, or
discarded control suffix.

When `end_padding=True`, fitting adds one equation at the endpoint
`chunk_size`, using a repeated final action. This is boundary supervision, not
future context: the caller still supplies exactly `chunk_size` actions.

For cubic `piper_chunk50`:

```text
executable actions       50
fitting input actions    50
directly fitted controls 28
discarded controls        0
span length               2 steps
```

The old extended behavior is available only when a loaded configuration
explicitly contains:

```yaml
implementation_version: uniform_left_extended_double_fit_v1
```

That compatibility path still consumes 56 Piper actions and reproduces the old
prefix fit. New configs default to `uniform_left_direct_fit_v1`. The two fit
procedures share the same decoding knot basis but generally produce different
controls. Their tokenizer fingerprints, fitted calibration bounds, and tokens
are intentionally not interchangeable. Retrain or regenerate preprocessing
artifacts when moving an existing model to the direct fitter.

### Adaptive left

Adaptive encoding fits one global episode spline using SciPy/FITPACK candidate
knots, optionally refits a repeated-action tail, and extracts local fixed-size
control prefixes through knot insertion.

```python
from spline_encoder import AdaptiveLeftBSplineConfig, create_encoder

encoder = create_encoder(AdaptiveLeftBSplineConfig(
    action_dim=7,
    chunk_size=20,
    frequency_hz=25,
    degree=3,
    num_basis=13,
    span_length_steps=2,
    fit_tolerance=1e-4,
    knot_insertion_excluded_dimensions=(6,),
))
records = encoder.encode_episode(episode_actions)
```

For each adaptive record:

```text
duration_steps.shape = (num_basis,)
executable_spans      = num_basis - degree
executable_steps      = sum(duration_steps[:executable_spans])
right-context spans   = degree
```

Honest limitations:

- Adaptive fitting is **episode-global and non-causal**. Each local record is
  derived from a spline fitted using the complete episode. It is not an online
  causal fitting algorithm.
- `fit_tolerance` guides FITPACK candidate selection on knot-driving
  dimensions. It is not a hard error bound for every dimension, local prefix,
  or the final tail-refitted curve.
- Excluded dimensions do not place knots, but are still fitted and stored.
  Discontinuous excluded channels such as grippers can have much larger error.
- `span_length_steps` guides density limiting and tail construction; adaptive
  durations are not forced to equal it.
- Native executable horizons vary. Repeating the final decoded action to fill a
  fixed tensor is a deployment convention, not an equivalent representation of
  missing future actions.
- Episodes with at most `degree` samples cannot produce adaptive records.
- Global adaptive fitting is slower than applying a cached uniform solver.

## Presets

Current presets use degree 3, vocabulary size 256, end padding, and span length
2.

| Preset | Action dim | Chunk | Frequency | Controls | Uniform-left input |
|---|---:|---:|---:|---:|---:|
| `libero` | 7 | 16 | 10 Hz | 11 | 16 |
| `libero_chunk8` | 7 | 8 | 10 Hz | 7 | 8 |
| `robotwin` | 14 | 16 | 30 Hz | 11 | 16 |
| `piper_chunk50` | 7 | 50 | 50 Hz | 28 | 50 |

These are geometry presets, not evidence that they are optimal for every
dataset associated with those benchmark names.

Custom classes are `UniformBSplineConfig`, `UniformDoubleBSplineConfig`,
`UniformLeftBSplineConfig`, `AdaptiveBSplineConfig`, and
`AdaptiveLeftBSplineConfig`.

## `SplineParameters` and units

| Field/property | Meaning |
|---|---|
| `control_points` | continuous fitted coefficients, `(num_basis, action_dim)` |
| `knots` | full knot vector in **seconds** |
| `knot_steps` | knots expressed in native action steps |
| `sample_period` | seconds per native action step |
| `executable_steps` | native endpoint-exclusive output length |
| `duration_steps` | integer distinct-knot gaps for left/adaptive layouts, otherwise `None` |
| `tokens` | optional quantized control points |
| `dequantized_control_points` | controls reconstructed from tokens |
| `observation_index` | source observation when encoded as an episode/dataset |
| `metadata` | layout annotations; not a promised stable extension schema |

```python
decoded = result.decode()
scipy_spline = result.scipy_spline()
decoded_quantized = result.decode(dequantized=True)  # only after quantization
rows = result.interleaved_control_durations()        # durations required
```

Passing `samples=` to `decode()` resamples across the executable time interval;
it does not simply truncate or pad native samples.

## Quantization

Quantization is optional and requires calibration:

```python
encoder.calibrate(training_windows)
result = encoder.encode_chunk(actions, quantize=True)
```

Current behavior:

- uniform scalar min/max bins with `vocab_size` levels;
- clipping outside calibrated bounds;
- uniform encoders calibrate per control row and action dimension;
- adaptive encoders calibrate per action dimension across local controls;
- constant calibration ranges decode to the constant bound;
- tokens are integer codes, not entropy-compressed bytes;
- durations are not quantized by `ControlQuantizer`.

Use representative **training** data for calibration. Calibrating on test data
is leakage. Min/max calibration is outlier-sensitive, and percentile control
calibration is not currently implemented.

`tokenizer_id` hashes configuration only; it does **not** include calibration
bounds. Two differently calibrated encoders can share an ID. Preserve and
compare the complete `encoder.json` when exact token compatibility matters.

## Serialization

```python
from spline_encoder import load_encoder, save_encoder

save_encoder(encoder, "encoder.json")
restored = load_encoder("encoder.json")
```

The state includes schema version, configuration, configuration fingerprint,
and optional calibration. The individual JSON write uses temporary-file atomic
replacement.

Validated dRTCv2 `action_tokenization` schema-v1/v2 Piper artifacts can be
loaded. This does not guarantee compatibility with every historical or
privately modified tokenizer JSON.

## LeRobot dataset export

The loader accepts local standard LeRobot data whose `meta/info.json` declares
`v2.1` or `v3.0`, with floating vector actions in parquet files.

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
```

`ActionAlignment` selects/reorders dimensions and can normalize fixed min/max
bounds to `[-1, 1]`. `infer_action_bounds()` computes dataset-wide quantiles.
Infer bounds on training data only, then reuse them for validation/test data.

The dataset `fps` must match the encoder frequency. Source timestamps are loaded
when present, but the encoder path currently assumes uniform timing from `fps`
and does not use irregular row timestamps. Use standalone `fit_bspline()` when
arbitrary timestamps must be honored.

Uniform episode tails repeat the final source action. Adaptive encoding fits the
complete episode and uses repeated-action knot support at its tail.

### Output

```text
output/
├── encoder.json
├── manifest.json
└── data/
    ├── episode_000000.npz
    └── ...
```

Each NPZ contains:

- `episode_index`, `frame_index`, `observation_index`
- `controls[observation, basis, action_dimension]`
- `knots[observation, basis + degree + 1]`
- `executable_steps[observation]`
- `duration_steps[observation, basis]` for left/adaptive layouts
- `control_duration_interleaved[..., action_dim + 1]` only for adaptive mode,
  with rows `[control dimensions..., duration]`
- `tokens` and `dequantized_controls` only when quantization is enabled

The source dataset is not modified. However:

- `overwrite=False` rejects an existing output directory;
- `overwrite=True` recursively deletes the resolved output directory first;
- directory export is not transactional, so interruption can leave partial
  output;
- the loader materializes parquet data/episodes in memory and is not streaming;
- the manifest fingerprint covers aligned frame indices and actions, not the
  complete dataset, videos, metadata, or environment;
- the package writes sidecars but has no high-level sidecar reader yet.

CLI example:

```bash
spline-encode /datasets/piper_pick encoded_splines/piper_pick \
  --preset piper_chunk50 \
  --mode uniform_left \
  --action-indices 0,1,2,3,4,5,6 \
  --normalization q01_q99
```

Use `--quantize --calibrate` for token export, `--encoder-state` to reuse a
saved state, and `--overwrite` only after checking the target path.

## Knot insertion and subdivision

```python
from spline_encoder import extract_interval, insert_knot, subdivide

spline = result.scipy_spline()
refined = insert_knot(spline, value=0.5)
left, right = subdivide(spline, value=0.5)
local = extract_interval(spline, left=0.2, right=0.8)
```

Boehm knot insertion preserves the represented curve up to floating-point
rounding. It does not preserve a quantized token layout.

```python
from spline_encoder import subdivide_parameters

left_result, right_result = subdivide_parameters(result, step=25)
```

High-level subdivision deliberately drops tokens because the control geometry
changes. `subdivide_parameters()` is documented for double-clamped results;
blind use on adaptive/left layouts does not preserve their duration metadata.

## Verification and visualization

```python
from spline_encoder import verify_reconstruction, verify_spline_parameters

invariants = verify_spline_parameters(result)
report = verify_reconstruction(
    result,
    reference_actions,
    max_absolute_error=0.05,
)
```

Invariant verification checks knot/control cardinality, monotonic knots, zero
origin, left clamping, domain coverage, partition of unity, finite decoding,
and duration/knot round-trip where applicable.

If `max_absolute_error` is omitted, `verify_reconstruction(...)["passed"]`
only indicates structural invariants passed. It does not assert acceptable
reconstruction accuracy.

```python
from spline_encoder import save_reconstruction_plot

save_reconstruction_plot(
    result,
    "verification/chunk.png",
    reference_actions,
    dimensions=(0, 1, 6),
)
```

Plots show targets, dense continuous/dequantized curves, knots, and control
coefficients at Greville abscissae. Control coefficients are generally not
points the curve passes through. A plausible plot is not proof of correctness.

One-window CLI:

```bash
python scripts/verify_chunk.py action_window.npy \
  --preset piper_chunk50 --mode uniform_left \
  --max-error 0.05 \
  --report verification/chunk.json \
  --plot verification/chunk.png
```

The input must exactly match `(input_chunk_size, action_dim)`. Without
`--max-error`, the CLI reports metrics but imposes no task-specific threshold.

## Tests and evidence

```bash
conda run -n starVLA python -m pytest -q
```

Most recent local results on 2026-09-15:

- 71 BSplineEncoder tests passed;
- 9 parent StarVLA B-spline training/deployment contract tests passed;
- known controls were recovered for degrees 1–5 in full-rank unregularized
  synthetic cases;
- design/evaluation agreed with SciPy;
- weighted ridge fits matched an independently assembled augmented system;
- randomized subdivision checks passed for degrees 1–5;
- quantization, serialization, LeRobot export, CLI, and PNG checks passed;
- all 34,254 cached Piper observations passed dRTCv2 parity for
  `uniform_double`, legacy `uniform_left_extended_double_fit_v1`, and
  `adaptive_left`, with zero control, knot, duration, token, cache-key, or
  coverage mismatches. Dequantized decode differences were approximately
  `1e-14` or smaller. This evidence does **not** apply to the new direct
  uniform-left controls, which intentionally implement a different fit.

Reproduce parity when the original repository/data are available:

```bash
python scripts/verify_original_tokenizer.py \
  --drtcv2-root /scratch/wangpc/dRTCv2 \
  --dataset /scratch/wangpc/dRTCv2/data/20260818_piper_pick_white_block_50hz \
  --report verification/piper_parity_report.json
```

Validate a sidecar through full recomputation:

```bash
python scripts/validate_lerobot_encoding.py SOURCE_DATASET ENCODED_SIDECAR \
  --output validation.json
```

These checks are not formal verification and do not prove control stability,
safety, task success, universal legacy compatibility, or good performance on
unseen datasets. The project has no published large-scale performance, memory,
GPU, multiprocessing, or thread-safety benchmark. Parent-repository tests also
have unrelated optional dependencies such as `pyzmq`.

## Repository layout

```text
src/spline_encoder/
├── adaptive.py       # adaptive FITPACK fitting
├── config.py         # configurations and presets
├── encoder.py        # observation-aligned encoders
├── fitting.py        # standalone control-point fitting
├── lerobot.py        # dataset input/output
├── quantization.py   # scalar quantization
├── result.py         # result dataclasses
├── serialization.py # JSON state
├── subdivision.py    # knot insertion/subdivision
├── verification.py  # numerical reports
└── visualization.py # optional plots
```

## License

MIT. See `LICENSE`.

---

# 中文

## 项目简介

BSplineEncoder 是一个尽量独立于训练框架的 Python 库，用 B 样条表示多维机器人动作
序列。它包括：

- 根据动作、时间戳和指定节点向量拟合控制点的基础接口；
- observation-aligned 固定节点和自适应动作编码器；
- 可选的控制点校准与量化；
- 本地 LeRobot v2.1/v3.0 数据集导出；
- 节点插入、曲线分段、数值验证和可视化；
- 显式加载当前 workspace 中已验证 dRTCv2 Piper uniform-left 旧 artifact 的兼容路径。

当前版本为 `0.1.0`，状态是 **Alpha**。API 可以使用，但尚未冻结。这是研究和预处理
代码，不是经过安全认证的控制器。

> **最重要的诚实说明：**控制点和节点向量保存的是拟合后的样条，不保证无误差保存
> 原始离散动作。一般拟合会产生近似误差，量化还会增加误差。用于策略之前，请在自己
> 的真实数据上测量重建质量。

## 安装

核心依赖为 Python `>=3.10`、NumPy `>=1.23` 和 SciPy `>=1.15`。

```bash
pip install "spline-encoder @ git+https://github.com/outsider86/BSplineEncoder.git"
```

开发时可克隆独立仓库并使用 editable install。可选依赖不会进入核心安装：

```bash
git clone https://github.com/outsider86/BSplineEncoder.git
cd BSplineEncoder
pip install -e .
pip install -e ".[lerobot]"       # pandas、PyArrow
pip install -e ".[visualization]" # Matplotlib
pip install -e ".[test]"          # 完整测试依赖
```

本项目已在 workspace 的 `starVLA` Conda 环境和 Python 3.10 下运行。仓库目前没有
完整 CI matrix 来证明所有 Python、NumPy、SciPy、系统和硬件组合都经过测试。

## 数学约定

对于 degree `k`、完整节点向量 `t` 和 `n` 个控制点 `C`：

```text
len(t) = n + k + 1
S(x)   = B(x; t, k) @ C
domain = [t[k], t[n]]
```

形状约定：

```text
timestamps     (num_samples,)
actions        (num_samples, action_dim)
design matrix  (num_samples, num_basis)
control points (num_basis, action_dim)
```

全部动作维度共享同一套时间戳和节点几何，并通过一次多输出矩阵最小二乘同时求解。
这里的“并行”不是为每个维度创建独立进程或线程。

## 使用指定节点拟合控制点

```python
import numpy as np
from spline_encoder import (
    evaluate_bspline,
    fit_control_points,
    make_clamped_knots,
)

timestamps = np.linspace(0.0, 2.0, 41)
actions = np.stack(
    (np.sin(timestamps), np.cos(timestamps), 0.25 * timestamps),
    axis=1,
)

# 输入是严格递增的 distinct span boundary；函数会重复端点，生成完整节点向量。
knots = make_clamped_knots([0.0, 0.5, 1.0, 1.5, 2.0], degree=3)
controls = fit_control_points(actions, timestamps, knots, degree=3)
reconstructed = evaluate_bspline(controls, knots, timestamps, degree=3)

assert controls.shape == (7, 3)
assert reconstructed.shape == actions.shape
```

`make_clamped_knots()` 接收互不重复、严格递增的分段边界；`fit_control_points()`
接收已经展开的完整节点向量。控制点数量始终等于
`len(knots) - degree - 1`。

需要诊断信息和重复求值时使用 `fit_bspline()`：

```python
from spline_encoder import fit_bspline

fit = fit_bspline(
    actions,
    timestamps,
    knots,
    degree=3,
    weights=None,
    regularization=1e-6,
)

print(fit.control_points)
print(fit.design_rank, fit.condition_number)
print(fit.rmse, fit.max_absolute_error)
query_times, query_actions = fit.sample(200)
```

求解目标为：

```text
min_C ||W (B C - A)||² + λ ||C||²
```

实现使用增广最小二乘，而不是显式计算 normal equations。输入必须有限；拟合时间戳
必须严格递增并位于定义域内；权重必须为正；degree 必须是 1–5 的整数。

默认 `require_full_rank=True` 会拒绝控制点不唯一的拟合。只有明确需要最小范数解或
正则化欠定解时才设置为 `False`。满秩也不等于数值稳定，应检查
`condition_number`。正则化能改善稳定性，但会使控制点产生偏差。

相关接口还有 `make_uniform_clamped_knots()`、`make_uniform_left_clamped_knots()`、
`bspline_design_matrix()`、`evaluate_bspline()` 和 `fit_bspline_control_points`。完整示例位于
`examples/fit_control_points.py`。

## Observation-aligned 编码器

```python
import numpy as np
from spline_encoder import create_encoder, preset

config = preset("piper_chunk50", mode="uniform_left")
encoder = create_encoder(config)

# direct uniform-left 直接用同一段 50 个动作拟合 50 个可执行动作。
actions = np.load("action_window.npy")
assert actions.shape == (config.input_chunk_size, config.action_dim)

result = encoder.encode_chunk(actions)
assert result.control_points.shape == (28, 7)
assert result.knots.shape == (32,)
assert result.decode().shape == (50, 7)
```

`encode_action_chunk()` 是单次调用封装。重复 uniform 拟合时应复用 encoder，因为
solver 已缓存。

### 模式

| 模式 | 输入 | 保存内容 | 原生输出 |
|---|---|---|---|
| `uniform_double` | 恰好 `chunk_size` 个动作 | 固定双端 clamped 节点和控制点 | 恰好 `chunk_size` 个采样 |
| `uniform_left` | 恰好 `chunk_size` 个动作 | 固定左端 clamped、右端开放节点和 `num_basis` 个控制点 | 恰好 `chunk_size` 个采样 |
| `adaptive_left` | 完整 episode；独立使用时为 `chunk_size` | 每条记录的控制点和整数 knot duration | 可变 horizon |

`uniform` 是 `uniform_double` 的历史别名。`adaptive` 与 `adaptive_left` 使用同一
实现；新代码使用明确的标准名称更清晰。

### Uniform double

设置 `span_length_steps` 时使用 endpoint-exclusive 几何：

```text
原生采样:      0 ... chunk_size - 1
样条定义域:    [0, chunk_size]
约束:          (num_basis - degree) * span_length_steps == chunk_size
```

默认 `end_padding=True` 会在拟合时把最后一个动作重复到右端点。
`span_length_steps=None` 则使用旧的 inclusive 布局。默认
`regularization=1e-4`，因此 encoder 是正则化近似器，不是严格插值器。

### Uniform left

`uniform_left` 现在直接构造左端 clamped、右端开放的节点向量，并直接从当前 action
chunk 拟合全部 `num_basis` 个控制点。设 span 长度为 `s`、degree 为 `k`、
`n = num_basis`，完整节点向量（动作步单位）是：

```text
[重复 k 次的 0] + [0, s, 2s, ..., n*s]
```

因此 0 的重数是 `k + 1`，SciPy 基础定义域为
`[0, (n-k)*s] = [0, chunk_size]`；最后 `k` 个节点提供右侧 basis support，但
不 clamp 右端。实现只构造一次这个矩阵，然后一次性直接求解全部控制点。没有扩展
action window、双端 clamped 中间样条、控制点前缀选择或丢弃尾部控制点。

`end_padding=True` 时，拟合会在 `chunk_size` 端点增加一个方程，目标值为重复的
最后一个动作。这是边界监督，不是 future context；调用者仍只提供 `chunk_size`
个动作。

三次 `piper_chunk50` 的实际布局：

```text
可执行动作       50
拟合输入动作     50
直接拟合控制点   28
丢弃控制点        0
span 长度         2 步
```

只有加载的配置显式包含以下字段时，才使用旧扩展拟合：

```yaml
implementation_version: uniform_left_extended_double_fit_v1
```

该兼容路径仍读取 56 个 Piper 动作，并复现旧 prefix fit。新配置默认使用
`uniform_left_direct_fit_v1`。两种算法的解码节点 basis 相同，但通常产生不同控制点；
tokenizer fingerprint、拟合得到的校准边界和 token 不能混用。已有模型迁移到 direct
fitter 时必须重新训练，或至少重新生成预处理和校准 artifact。

### Adaptive left

Adaptive 模式通过 SciPy/FITPACK 候选节点拟合一个全 episode 样条；可选地重新拟合
重复动作尾部；然后通过节点插入提取控制点数量固定的局部前缀。

```python
from spline_encoder import AdaptiveLeftBSplineConfig, create_encoder

encoder = create_encoder(AdaptiveLeftBSplineConfig(
    action_dim=7,
    chunk_size=20,
    frequency_hz=25,
    degree=3,
    num_basis=13,
    span_length_steps=2,
    fit_tolerance=1e-4,
    knot_insertion_excluded_dimensions=(6,),
))
records = encoder.encode_episode(episode_actions)
```

每条记录满足：

```text
duration_steps.shape = (num_basis,)
executable_spans      = num_basis - degree
executable_steps      = sum(duration_steps[:executable_spans])
right-context spans   = degree
```

必须明确的限制：

- Adaptive 拟合是**全 episode、非因果**过程。局部记录来自用完整 episode 拟合的
  样条，不是在线 causal fitting。
- `fit_tolerance` 只指导参与节点选择维度上的 FITPACK candidate selection，不是对
  每个维度、每条局部记录或 tail-refit 曲线的硬误差保证。
- 被 `knot_insertion_excluded_dimensions` 排除的维度不会放置节点，但仍会被拟合和
  保存。不连续的 gripper 等通道可能出现明显更大的误差。
- `span_length_steps` 指导密度上限和尾部构造，不强制 adaptive duration 相等。
- 原生可执行 horizon 会变化。重复最后一个解码动作来填满固定 tensor 是部署约定，
  不等价于样条表示了缺失的未来动作。
- 动作数不超过 `degree` 的 episode 无法产生 adaptive 记录。
- 全 episode adaptive 拟合比缓存 uniform solver 更慢。

## Preset

当前 preset 都使用 degree 3、vocabulary size 256、end padding 和 span length 2。

| Preset | 动作维度 | Chunk | 频率 | 控制点 | Uniform-left 输入 |
|---|---:|---:|---:|---:|---:|
| `libero` | 7 | 16 | 10 Hz | 11 | 16 |
| `libero_chunk8` | 7 | 8 | 10 Hz | 7 | 8 |
| `robotwin` | 14 | 16 | 30 Hz | 11 | 16 |
| `piper_chunk50` | 7 | 50 | 50 Hz | 28 | 50 |

这些只是几何 preset，不证明它们对所有同名 benchmark 数据都是最优选择。

## `SplineParameters` 与单位

| 字段/属性 | 含义 |
|---|---|
| `control_points` | 连续拟合系数，形状 `(num_basis, action_dim)` |
| `knots` | 单位为**秒**的完整节点向量 |
| `knot_steps` | 以原生动作步表示的节点 |
| `sample_period` | 每个原生动作步的秒数 |
| `executable_steps` | endpoint-exclusive 原生输出长度 |
| `duration_steps` | left/adaptive 布局的整数 distinct-knot 间隔 |
| `tokens` | 可选量化控制点 |
| `dequantized_control_points` | token 反量化后的控制点 |
| `observation_index` | episode/数据集中的源 observation |

```python
decoded = result.decode()
scipy_spline = result.scipy_spline()
decoded_quantized = result.decode(dequantized=True)
rows = result.interleaved_control_durations()
```

`decode(samples=...)` 会在可执行时间区间重新采样，不是简单截断或 padding。

## 量化

量化默认关闭，使用前必须校准：

```python
encoder.calibrate(training_windows)
result = encoder.encode_chunk(actions, quantize=True)
```

当前实现是 `vocab_size` 个 level 的标量 uniform min/max bin。超出范围的值会被
clip。Uniform encoder 按控制点行和动作维度校准；adaptive encoder 按动作维度在
全部局部控制点上校准。Token 是整数 code，不是压缩 byte stream；duration 不由
`ControlQuantizer` 量化。

应只使用有代表性的训练数据校准；使用 test 数据校准属于数据泄漏。Min/max 对 outlier
敏感，目前没有控制点 percentile calibration。

`tokenizer_id` 只哈希**配置**，不包含校准范围。不同校准的 encoder 可以拥有相同
ID。需要精确 token 兼容时，应保存并比较完整 `encoder.json`。

## 序列化

```python
from spline_encoder import load_encoder, save_encoder

save_encoder(encoder, "encoder.json")
restored = load_encoder("encoder.json")
```

State 保存 schema version、配置、配置 fingerprint 和可选校准。单个 JSON 文件使用
临时文件和 atomic replacement 写入。

经过验证的 dRTCv2 `action_tokenization` schema-v1/v2 Piper artifact 可以加载；这
不保证所有历史版本或私人修改 JSON 都兼容。

## LeRobot 数据集导出

Loader 支持 `meta/info.json` 声明为 `v2.1` 或 `v3.0`、并在 parquet 中存储浮点
向量动作的本地标准 LeRobot 数据集。

```python
from spline_encoder import ActionAlignment, create_encoder, encode_lerobot_dataset, preset

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
```

`ActionAlignment` 可以选择/重排维度，并使用固定 min/max 归一化到 `[-1, 1]`。
`infer_action_bounds()` 计算全数据集 quantile。应只从训练数据推断范围，再复用于
validation/test。

数据集 `fps` 必须与 encoder frequency 匹配。Loader 虽然会读取 source timestamp，
但当前 encoder 路径按 `fps` 假设均匀采样，不使用不规则逐行时间戳。需要任意时间戳时
应使用独立 `fit_bspline()`。

输出目录：

```text
output/
├── encoder.json
├── manifest.json
└── data/episode_XXXXXX.npz
```

NPZ 中包括：

- `episode_index`、`frame_index`、`observation_index`
- `controls`、`knots`、`executable_steps`
- left/adaptive 布局的 `duration_steps`
- 仅 adaptive 模式的 `control_duration_interleaved`，行布局为
  `[control dimensions..., duration]`
- 仅量化时的 `tokens` 和 `dequantized_controls`

源数据不会被修改，但必须注意：

- `overwrite=False` 会拒绝已有输出目录；
- `overwrite=True` 会先递归删除解析后的整个输出目录；
- 目录导出不是 transactional，中断可能留下部分结果；
- 当前 loader 会把 parquet/episode 放入内存，不是 streaming；
- manifest fingerprint 只覆盖对齐后的 frame index 和 action，不覆盖完整数据集、
  视频、metadata 或环境；
- 当前只有 sidecar writer，没有高级 sidecar reader。

CLI：

```bash
spline-encode /datasets/piper_pick encoded_splines/piper_pick \
  --preset piper_chunk50 \
  --mode uniform_left \
  --action-indices 0,1,2,3,4,5,6 \
  --normalization q01_q99
```

`--quantize --calibrate` 会校准并导出 token；`--encoder-state` 复用已有 state；使用
`--overwrite` 前必须确认路径。

## 节点插入和分段

```python
from spline_encoder import extract_interval, insert_knot, subdivide

spline = result.scipy_spline()
refined = insert_knot(spline, value=0.5)
left, right = subdivide(spline, value=0.5)
local = extract_interval(spline, left=0.2, right=0.8)
```

Boehm 节点插入会在浮点误差范围内保持曲线，但不会保留量化 token 布局。
`subdivide_parameters(result, step=...)` 面向 double-clamped 结果，并有意丢弃 token。
不应假设它能保留所有 left/adaptive duration metadata 语义。

## 验证与可视化

```python
from spline_encoder import verify_reconstruction, verify_spline_parameters

invariants = verify_spline_parameters(result)
report = verify_reconstruction(
    result,
    reference_actions,
    max_absolute_error=0.05,
)
```

验证内容包括节点/控制点数量、节点单调性、零起点、左端 clamping、定义域覆盖、
partition of unity、有限解码和 duration/knot round-trip。

如果省略 `max_absolute_error`，`passed` 只代表结构不变量通过，**不代表**重建精度满足
实际任务。

```python
from spline_encoder import save_reconstruction_plot

save_reconstruction_plot(
    result,
    "verification/chunk.png",
    reference_actions,
    dimensions=(0, 1, 6),
)
```

图中展示 target、连续/反量化曲线、节点和位于 Greville abscissa 的控制系数。控制
系数一般不是曲线必须经过的点；看起来合理的图不是正确性证明。

单窗口 CLI：

```bash
python scripts/verify_chunk.py action_window.npy \
  --preset piper_chunk50 --mode uniform_left \
  --max-error 0.05 \
  --report verification/chunk.json \
  --plot verification/chunk.png
```

输入必须严格匹配 `(input_chunk_size, action_dim)`。不提供 `--max-error` 时只报告
指标，不设置任务精度阈值。

## 测试证据

```bash
conda run -n starVLA python -m pytest -q
```

2026-09-15 最近一次本地结果：

- 71 个 BSplineEncoder 测试通过；
- 9 个上层 StarVLA B-spline 训练/部署 contract 测试通过；
- degree 1–5 的满秩、无正则化合成 case 能恢复已知控制点；
- design/evaluation 与 SciPy 一致；
- 加权正则化拟合与独立构造的增广系统一致；
- degree 1–5 的随机 subdivision 检查通过；
- 量化、序列化、LeRobot 导出、CLI 和 PNG 检查通过；
- `uniform_double`、旧版 `uniform_left_extended_double_fit_v1` 和 `adaptive_left` 的
  全部 34,254 条 Piper cache observation 均通过 dRTCv2 parity，control、knot、
  duration、token、cache key 和覆盖数 mismatch 都为零；反量化 decode 差异约为
  `1e-14` 或更小。该证据**不适用于**新的 direct uniform-left 控制点，因为它有意
  使用了不同的拟合定义。

```bash
python scripts/verify_original_tokenizer.py \
  --drtcv2-root /scratch/wangpc/dRTCv2 \
  --dataset /scratch/wangpc/dRTCv2/data/20260818_piper_pick_white_block_50hz \
  --report verification/piper_parity_report.json

python scripts/validate_lerobot_encoding.py SOURCE_DATASET ENCODED_SIDECAR \
  --output validation.json
```

这些测试不是 formal verification，不能证明控制稳定性、安全性、任务成功率、所有旧
artifact 兼容性或对未知数据的效果。项目目前没有公开的大规模性能、内存、GPU、多进程
或线程安全 benchmark。父仓库完整测试还需要 `pyzmq` 等与本包数值逻辑无关的可选依赖。

## 目录结构

```text
src/spline_encoder/
├── adaptive.py       # adaptive FITPACK 拟合
├── config.py         # 配置与 preset
├── encoder.py        # observation-aligned encoder
├── fitting.py        # 独立控制点拟合
├── lerobot.py        # 数据集输入输出
├── quantization.py   # 标量量化
├── result.py         # 结果 dataclass
├── serialization.py # JSON state
├── subdivision.py    # 节点插入/分段
├── verification.py  # 数值报告
└── visualization.py # 可选绘图
```

## License

MIT，详见 `LICENSE`。
