# BSplineEncoder implementation review

| | |
|---|---|
| Repository | <https://github.com/outsider86/BSplineEncoder> |
| Reviewed commit | `5aede68cee997a2d501210ce3dfbc15144d44447` (`main`, "Expand standalone B-spline encoder API"); working tree was clean |
| Review date | 2026-10-15 |
| Result state | Committed on top of `5aede68` on branch `review-reorganize-piper-pipeline` and pushed to `origin` for review; not merged into `main`; no package published |
| Intended reader | An independent second reviewer |

Passing tests do **not** establish that the implementation is error-free. This
review targeted specific numerical contracts with independent references. Its
evidence covers one software environment and synthetic data plus one
real-robot parity dataset. The [unresolved limitations](#7-documented-limitations-and-unresolved-items)
list what remains unverified.

## 1. Scope

In scope, audited by reading the source and probing numerically:

- **Fitting procedures:**
  - `uniform_double`/`uniform`, in both inclusive (`span_length_steps=None`)
    and endpoint-exclusive layouts;
  - direct `uniform_left` (`uniform_left_direct_fit_v1`);
  - legacy extended `uniform_left_extended_double_fit_v1`, plus the historical
    v2 version string;
  - `adaptive_left`/`adaptive`, including FITPACK candidate selection, the
    repeated-tail refit, and per-observation prefix extraction.
- **Geometry and decoding:** knot/control counts; seconds versus steps;
  endpoints and endpoint padding; duration alignment and the interleaved layout.
- **Quantization, serialization, and fingerprints.**
- **LeRobot observation alignment:** short episodes and tail windows.
- **Geometry operations:** `insert_knot`, `subdivide`, `extract_interval`,
  `extract_left_clamped_prefix`, `subdivide_parameters`.
- **Verification and visualization:** `verify_spline_parameters`,
  `verify_reconstruction`, `plot_reconstruction`, examples, and scripts.

Out of scope: policy training, GPU experiments, performance benchmarking,
thread safety, the parent StarVLA/dRTCv2 code (used only as the parity
oracle), and anything under `src/data/`. That directory appeared, untracked,
in the working tree during the review. It is not part of the reviewed commit
and was not modified during this review. Those supplied scripts now live
byte-identically in `scripts/legacy_piper/`, and `src/data/` is the `data`
package (§11).

## 2. Environment and exact test results

| Item | Value |
|---|---|
| OS | Linux 6.8.0-146-generic x86_64, glibc 2.39 |
| Python | 3.10.19 (conda env `starVLA`) |
| NumPy / SciPy | 1.26.4 / 1.15.3 |
| matplotlib / pandas / pyarrow / h5py / pytest | 3.10.8 / 2.3.3 / 14.0.1 / 3.16.0 / 9.1.1 |
| Hardware use | CPU only |

> **Import-path caveat.** The `starVLA` environment holds an editable install
> of `spline-encoder` that points to a *different* checkout,
> `/scratch/wangpc/DiscreteRTCv2/BSplineEncoder`, also at `5aede68`. All runs
> below therefore force this checkout with `PYTHONPATH=src`; the
> subprocess-based tests inherit it. Without that, tests silently exercise the
> other checkout.

Command (run from the repository root):

```bash
PYTHONPATH=src /home/wangpc/miniconda3/envs/starVLA/bin/python -m pytest -q -p no:cacheprovider -rs
```

| Run | Passed | Failed | Skipped | Notes |
|---|---:|---:|---:|---|
| Baseline `5aede68`, before any change | 71 | 0 | 0 | 21 pandas `DeprecationWarning`s (BlockManager), unrelated |
| After the review | **172** | **0** | **2** | same 21 warnings; ~5.6 s |
| New review tests run against the *baseline* source | 83 | **18** | 2 | demonstrates that the tests detect the defects (the visual-script smoke test always imports this checkout's `src`, so it is not a baseline discriminator) |
| Whole suite against the *baseline* source | 154 | 18 | 2 | the same 18 |

The two skips are mathematically inapplicable parameter combinations, not
untested behavior:

- `test_reference_geometry.py:74`: degree-1 insertion at an existing simple
  knot would exceed the supported multiplicity (`< degree`).
- `test_reference_geometry.py:116`: a double interior knot needs degree ≥ 2.

Collected tests per file after the review: `test_core_fitting` 14,
`test_encoder` 12, `test_lerobot` 6, `test_numerical_properties` 16,
`test_prefix_transfer` 25 (new), `test_public_api` 2,
`test_reference_geometry` 50 (new), `test_review_regressions` 28 (new),
`test_subdivision` 2, `test_validation_contracts` 12, `test_verification` 7.

The 18 baseline failures map to findings F1 (2 tests), F2 (1), F3 (10),
F4 (3), F5 (1), and F7 (1).

## 3. Findings summary

Severity scale:
- **High:** silently wrong numbers on a default or encoder path.
- **Medium:** silently wrong or non-finite numbers from a public API on valid
  but non-default inputs.
- **Low:** wrong representation or metadata with correct values, a crash
  instead of a report, or a broken example.

No High-severity defect was found. The encoder paths whose outputs feed
datasets and policies (all fixed-knot modes and adaptive episode encoding)
are bit-for-bit unchanged by the fixes ([§6](#6-compatibility-evidence)).

| ID | Severity | Status | Component | Summary |
|---|---|---|---|---|
| F1 | Medium | Fixed | `extract_left_clamped_prefix` | Silently wrong curve for repeated interior knots and for open-left inputs cut at the domain start (hypothesis 2: **confirmed**) |
| F2 | Medium | Fixed | `subdivide_parameters` | Inclusive layout: right child's last native sample decoded to NaN in 29.6% of splits; last step passed the guard (hypothesis 1: **confirmed**, by a different mechanism than proposed) |
| F3 | Low | Fixed | `extract_interval` | Open boundaries at the original endpoints were returned unclamped despite the "double-clamped" contract; values were correct (hypothesis 3: **confirmed**) |
| F4 | Medium | Fixed | Uniform encoders | `regularization=0` with a rank-deficient basis silently produced controls of about 2.5e18 |
| F5 | Low | Fixed | `verify_spline_parameters` | Crashed on wrong-length durations; passed an endpoint-exclusive horizon extending past the domain |
| F6 | Low | Fixed | `examples/encode_chunk.py` | Passed 56 actions to the 50-action direct `uniform_left` preset and crashed |
| F7 | Low | Fixed | `subdivide_parameters` | Left child of a `uniform_left` result was labelled `left_clamped_right_open` although it is double-clamped |

Hypotheses and claims that were **disproved**:

| Claim | Verdict | Evidence |
|---|---|---|
| H1 as stated: copying `inclusive_domain` to children misaligns their native sample *indices* | Disproved | Children cover native samples `0..s-1` and `s..N-1` exactly. The actual defect (F2) is a floating-point domain-end shortfall, not the metadata copy. |
| H2 affects encoder outputs | Disproved | Every global adaptive spline in the probes and tests was left-clamped with simple interior knots; prefixes reproduced it to ≤ 5.4e-15. The parity rerun shows 0 control/knot/duration mismatches over 34,254 observations. |
| The cached normal-equation uniform solvers are inaccurate for shipped presets | Disproved | They agree with stable augmented LAPACK least squares to ≤ 4.3e-14 for every preset and mode (see L1). |
| `k` complete control/duration pairs need all `k` durations to fix the first `k-p` spans | Disproved (stronger result) | Span `m` depends only on knots `u_{m-p+1}..u_{m+p}`, so the `k`-th duration only shapes right-side support (see L3). |

## 4. Confirmed defects: reproducers, fixes, and downstream impact

Every reproducer below runs against `5aede68` with
`PYTHONPATH=<baseline>/src`, where `<baseline>` comes from
`git archive 5aede68 src | tar -x -C <baseline>`.

### F1. `extract_left_clamped_prefix` silently misaligns (hypothesis 2)

```python
import numpy as np
from scipy.interpolate import BSpline
from spline_encoder import extract_left_clamped_prefix

# (a) repeated interior knot (2.0 twice) inside the retained window
knots = np.r_[[0.] * 4, 1, 2, 2, 3, 4, 5, 6, 7, [8.] * 4]
s = BSpline(knots, np.random.default_rng(0).normal(size=(13, 2)), 3)
p = extract_left_clamped_prefix(s, 0.5, num_knot_spans=6)
x = np.linspace(0, p.distinct_knots[p.num_executable_spans], 200)
np.abs(p.executable_spline()(x) - s(x + 0.5)).max()      # baseline: 0.906

# (b) open-left (non-clamped) input cut at its domain start
s = BSpline(np.arange(-3., 13.), np.random.default_rng(1).normal(size=(12, 2)), 3)
p = extract_left_clamped_prefix(s, 0.0, num_knot_spans=5)
x = np.linspace(0, p.distinct_knots[p.num_executable_spans], 200)
np.abs(p.executable_spline()(x) - s(x)).max()            # baseline: 1.479
```

**Cause.** The prefix uses `np.unique(knots)` and assumes one control per
distinct span, so a repeated interior knot shifts every later control/knot
pairing. At `left == domain start` the code skipped subdivision, assuming the
input was already clamped.

**Fix** (`subdivision.py`):
- An open left boundary is now clamped via `_clamped_piece`, Boehm insertion
  at the boundary. Already-clamped inputs pass through with identical arrays.
- Repeated knots strictly inside `(left, last retained break)` raise
  `ValueError`. One duration per control cannot encode them, so this enforces
  the supported domain rather than guessing. A repeated knot exactly at `left`
  remains supported, because subdivision clamps it.

**Impact.** Only direct callers of this public function with such inputs
received wrong prefixes. The adaptive encoder never produces these inputs
(disproved row above), so stored adaptive data is unaffected.

Regression tests:
- `test_prefix_transfer.py::test_prefix_of_open_left_input_is_clamped_even_at_the_domain_start`
- `test_prefix_transfer.py::test_prefix_rejects_repeated_interior_knots_instead_of_misaligning`

### F2. `subdivide_parameters` on the inclusive layout (hypothesis 1)

```python
import numpy as np
from spline_encoder import UniformDoubleBSplineConfig, create_encoder, subdivide_parameters
cfg = UniformDoubleBSplineConfig(action_dim=2, chunk_size=17, frequency_hz=10.0,
                                 num_basis=11, span_length_steps=None)
t = np.linspace(0, 2, 17); r = create_encoder(cfg).encode_chunk(np.stack([np.sin(t), np.cos(t)], 1))
L, R = subdivide_parameters(r, 12)
R.knots[-1], (R.executable_steps - 1) * R.sample_period   # 0.3999999999999999, 0.4
np.isnan(R.decode()).any()                                  # baseline: True
subdivide_parameters(r, 16)   # passes the guard, then fails inside subdivide()
```

**Cause.** The right child's knots are shifted as `t - split_time` in seconds.
`end - split_time` can round one ulp below the inclusive last-sample time
`(steps - 1) * sample_period`, which SciPy (`extrapolate=False`) then
evaluates to NaN.

A sweep of 7,974 splits (6 frequencies × chunk 8–32 × `num_basis` ∈ {6, 8,
11}) found **2,358 NaN-producing splits at baseline and 0 after the fix**.
Separately, `step = executable_steps - 1` lies on the domain end for inclusive
layouts but passed the up-front check.

**Fix:**
- For inclusive parents, right-child knots equal to the parent domain end are
  snapped to the exact inclusive horizon. This applies only when the two agree
  within `1e-9 · sample_period`, so the change is at most an ulp-level knot
  adjustment.
- The step guard now also requires the split time to be strictly inside the
  spline domain, raising the original message.
- Endpoint-exclusive layouts are untouched.

**Impact.** Before the fix, callers subdividing inclusive results could receive
a NaN final action, which `verify_spline_parameters` flagged as failing.
Encoders never call `subdivide_parameters`, so stored encodings are
unaffected.

**Not changed** (documented as L8): the left child of an inclusive parent keeps
`inclusive_domain=True`. Its native samples are correct, but
`decode(samples=M)` resamples `[0, (s-1)Δt]` rather than the piece's full
`[0, sΔt]`.

Regression test:
`test_review_regressions.py::test_subdivide_parameters_children_stay_aligned_with_native_samples[None]`,
which checks every valid split.

### F3. `extract_interval` leaves open boundaries unclamped (hypothesis 3)

```python
import numpy as np
from spline_encoder import create_encoder, preset, extract_interval
r = create_encoder(preset("libero_chunk8", "uniform_left")).encode_chunk(np.random.default_rng(0).normal(size=(8, 7)))
iv = extract_interval(r.scipy_spline(), 0.2, 0.8)   # 0.8 s is the right-open domain end
iv.t   # baseline: [0 0 0 0 .2 .4 .6 .8 1. 1.2] -> local right end 0.6 has multiplicity 1; support knots leak in
       # fixed:    [0 0 0 0 .2 .4 .6 .6 .6 .6]
```

An open-uniform input (`knots = arange(-3, 13)`) was returned completely
unchanged by `extract_interval(s, 0, 9)`.

Values on `[0, right-left]` were always correct (error ≤ 5.6e-16), but the
docstring's "double-clamped" contract was violated. Consumers that read
endpoint values from the end controls, concatenate pieces, or count `degree +
1` boundary knots would be wrong.

**Fix.** Boundaries that coincide with an open end of the input are clamped by
`_clamped_piece`. Interior cuts still use the unchanged `subdivide`, and
clamped inputs give bit-identical results.

Regression tests: `test_reference_geometry.py::test_extract_interval_is_double_clamped_at_every_boundary[*]`
covers clamped, right-open, and open-uniform inputs at degrees 1–5. It checks
multiplicities, values, and one-sided derivatives up to order `p`.

### F4. Unregularized rank-deficient uniform solver returns huge controls

```python
from spline_encoder import UniformDoubleBSplineConfig, create_encoder
import numpy as np
enc = create_encoder(UniformDoubleBSplineConfig(action_dim=1, chunk_size=4, num_basis=8,
                                                span_length_steps=None, regularization=0.0))
c = enc.encode_chunk(np.arange(4.)[:, None]).control_points
np.abs(c).max()   # baseline: 2.46e18, decode error 77, no error raised
```

Configuration validation accepts `num_basis` larger than the number of fit
samples. With `regularization=0`, `np.linalg.inv(BᵀB)` (or `solve`) of a
singular matrix returned finite garbage.

**Fix.** At encoder construction:
- `_check_fit_basis` raises `ValueError` when `regularization == 0` and the
  fit basis is column-rank deficient.
- `_check_solver` rejects a non-finite solver.

No numerics change for valid configurations: the checks run around the
unchanged solver expressions.

**Impact.** Not reachable with presets or the default `regularization=1e-4`.

Regression test:
`test_review_regressions.py::test_unregularized_rank_deficient_uniform_layout_is_rejected[*]`.

### F5. Malformed-result verification gaps

```python
from dataclasses import replace
import numpy as np
from spline_encoder import create_encoder, preset, verify_spline_parameters
actions = np.stack([np.sin(np.linspace(0, 3, 40) + i) for i in range(7)], 1)
rec = create_encoder(preset("libero_chunk8", "adaptive_left")).encode_episode(actions)[5]
verify_spline_parameters(replace(rec, duration_steps=rec.duration_steps[:-1]))  # baseline: ValueError (broadcast)
bad = replace(rec, executable_steps=int(round(rec.knot_steps[rec.num_basis])) + 1)
verify_spline_parameters(bad)["passed"]      # baseline: True, yet bad.decode(samples=200) contains NaN
```

**Fix:**
- A wrong-length duration vector now yields
  `checks["duration_knot_round_trip"] = False`.
- A new additive check, `executable_horizon_inside_domain`, requires the
  endpoint-exclusive horizon `executable_steps · sample_period` (or the last
  sample for inclusive layouts) to lie inside the spline domain.

All encoder outputs, both children of every valid subdivision, and all prior
tests still pass verification. The returned report keeps its keys and gains
one check key.

Regression tests: `test_review_regressions.py::test_verification_reports_rather_than_crashes_on_malformed_results`
and `::test_encoder_outputs_satisfy_the_horizon_check_and_unit_conventions[*]`.

### F6. Broken example

`examples/encode_chunk.py` built a `(56, 7)` window for the direct
`uniform_left` Piper preset, which requires `(50, 7)`, so it raised
`ValueError`. It now uses `config.input_chunk_size`. Regression test:
`test_review_regressions.py::test_offline_examples_run[*]`.

### F7. Left child metadata

`subdivide_parameters` copied `boundary_condition="left_clamped_right_open"`
to the left child of a `uniform_left` result, although subdivision always
double-clamps the cut. The left child is now labelled `double_clamped`; the
right child keeps the parent's label. This is metadata only and no
numbers change. Regression test:
`test_review_regressions.py::test_subdivided_uniform_left_left_child_is_labelled_double_clamped`.

## 5. Independent-reference verification (beyond round trips)

| Property | Reference | Coverage | Worst observed |
|---|---|---|---|
| Basis evaluation (`_design`, `bspline_design_matrix`) | `scipy.interpolate.BSpline.design_matrix` | degrees 1–5; nonuniform clamped and right-open knots; samples at every knot and at the domain end | ≤ 3.3e-16 (exact for the public helper) |
| Knot insertion (`insert_knot`) | FITPACK `scipy.interpolate.insert` | degrees 1–5; between-knot and existing-knot insertion; nonuniform knots; per action dimension | identical knot vectors; coefficients ≤ 2.2e-16 |
| Weighted fitting (`fit_bspline`) | LAPACK `gelsd` on the weighted design | degrees 1–5; nonuniform clamped and right-open knots; known-spline recovery | ≤ 1e-13·cond |
| Cached uniform solvers | augmented `gelsd` least squares, `[B; √λ I]` | 4 presets × {double, direct left, legacy left} | ≤ 4.3e-14 |
| Subdivision | `PPoly.from_spline` one-sided limits | degrees 1–5; cuts between knots, at simple knots, and at double knots; derivative orders 0..p | value ≤ 2e-13; pieces C^(p−m) across the cut |
| Interval extraction | source spline, `PPoly` | degrees 1–5; clamped, right-open, and open-uniform inputs; all boundary combinations | ≤ 5.6e-16 |
| Prefix: `k` pairs → `k−p` spans | directly built left-clamped splines | degrees 1–5; every `k` from `p+1` to `p+9` | ≤ 1e-13; tightness asserted |
| Suffix-to-prefix transfer | source spline | degrees 1–5; existing-knot, between-knot, and domain-start cuts; adaptive records; Piper `uniform_left` | ≤ 1e-12 (figure run: 4.4e-16) |
| Requantization after transfer | half-bin bound; source control hull | adaptive, vocab 64 and 256 | ≤ half-bin per dimension; 0 clipped controls |

Key numerical facts behind the tests:

- **Committed-interval invariance.** For degree `p` and a left-clamped knot
  vector `[0]*(p+1) + u_1..u_k`, span `m` uses controls `m..m+p` and knots
  `u_{m-p+1}..u_{m+p}`. Therefore `k` controls plus the first `k-1` durations
  determine spans `0..k-p-1` exactly. Perturbing controls `≥ k` or durations
  `≥ k` leaves them unchanged, to ≤ 1e-13. Perturbing control `k-1` or duration
  `k-1` does move the last committed span, so the bound is tight.
- **Transfer and quantization.** Boehm controls are convex combinations of
  source controls. A transferred prefix therefore stays inside the per-dimension
  calibration hull of the source record, and adaptive requantization never
  clips. Requantization error is bounded by one half-bin per dimension.
  Transferring an already dequantized record and requantizing it again is
  bounded by two half-bins. Continuous transfer error is measured separately
  and is at the rounding level.
- **Seconds versus steps.** `knots == knot_steps · Δt` to 1e-15 for every mode.
  Adaptive `np.unique(knot_steps)` equals `cumsum(duration_steps)`, and
  `executable_steps = distinct[num_basis − degree]` (asserted in tests).

## 6. Compatibility evidence

1. **dRTCv2 Piper parity rerun on the reviewed code.** Data:
   `/scratch/wangpc/dRTCv2/data/20260818_piper_pick_white_block_50hz`, 52
   episodes, 34,254 observations.

   ```bash
   python scripts/verify_original_tokenizer.py --drtcv2-root /scratch/wangpc/dRTCv2 \
     --dataset /scratch/wangpc/dRTCv2/data/20260818_piper_pick_white_block_50hz \
     --report <out>.json
   ```

   | Mode | Fingerprint | Control / knot max diff | Token / duration / cache mismatches | Dequantized decode max diff |
   |---|---|---|---|---|
   | `uniform_double` | `b2211e5e8b95d867` | 0.0 / 0.0 | 0 / – / 0 | 1.28e-15 |
   | `uniform_left` legacy extended | `3f7924fa97a514c5` | 0.0 / 0.0 | 0 / – / 0 | 0.0 |
   | `adaptive_left` | `88cf71f30492871b` | 0.0 / 0.0 | 0 / 0 / 0 | 9.33e-15 |

   These match the committed `verification/piper_parity_report.json` exactly
   (`passed: true`, 135.6 s). Parity does not cover direct `uniform_left`,
   which intentionally differs from dRTCv2.

2. **Bit-exact before/after snapshot.** `scripts/compare_output_snapshots.py`
   compared 546 arrays (807,784 values), with zero bitwise differences. The
   snapshot covers:
   - all presets × all modes, plus legacy `uniform_left`;
   - inclusive, left, and adaptive layouts at degrees 1–5;
   - adaptive with an excluded dimension;
   - controls, knots, durations, tokens, dequantized controls, continuous and
     dequantized decodes, and `encoder.decode(tokens)`;
   - fingerprints and serialized `encoder.json` state;
   - exclusive-layout subdivision children, `extract_interval` on clamped
     inputs, and left prefixes.

   Commands are in the [README](../README.md#tests-and-evidence). Each snapshot
   prints the module path it imported, so the two sides are provably different
   checkouts.

3. **Public surface.** No public import, signature, default, return type or
   shape, unit, configuration field, CLI flag, or serialization schema
   changed. Additions are private helpers (`_boehm_insert`, `_clamped_piece`,
   `_check_fit_basis`, `_check_solver`) and one new key in the
   `verify_spline_parameters()["checks"]` dict.
   `insert_knot_once` keeps its validation; its arithmetic moved verbatim into
   `_boehm_insert`.

4. **Optional dependencies.** The core imports and encodes with `matplotlib`,
   `pandas`, `pyarrow`, and `h5py` blocked via `sys.modules`. Only the scripts
   and the matplotlib-dependent test use plotting; that test calls
   `pytest.importorskip`.

**Behavior changes visible to callers (corrections):**
- `extract_left_clamped_prefix` raises for repeated interior knots, and clamps
  open-left inputs.
- `extract_interval` clamps open ends, so it can return a different control
  and knot count for such inputs.
- The inclusive right child of `subdivide_parameters` may have its final knots
  moved by about one ulp, and `step = executable_steps - 1` on the inclusive
  layout now raises the documented message earlier.
- Unregularized rank-deficient uniform configurations raise at construction.
- Verification can report `passed: False` for results that previously passed
  but were malformed.

## 7. Documented limitations and unresolved items

These are research choices or limitations, not defects. None were changed.

- **L1 — Normal-equation solvers.** `UniformBSplineEncoder` caches
  `inv(BᵀB + λI) Bᵀ`, and the left encoders use `solve`. Their accuracy:

  | Fit | Condition number | Agreement with stable least squares |
  |---|---|---|
  | Shipped presets (`λ = 1e-4`) | `cond(BᵀB+λI)` ≤ 543 | ≤ 4.3e-14 |
  | Direct uniform-left, `λ = 0`, degree 4 | `cond(B)` = 222 | 6.9e-12 |
  | Direct uniform-left, `λ = 0`, degree 5 | `cond(B)` = 3.1e3 | 3.6e-10 |

  The direct uniform-left basis is worse conditioned (`cond(B)` ≈ 23 versus
  ≈ 4.6 for the double layout at degree 3) because its last controls are only
  weakly observed inside `[0, chunk]`. The cached form was kept because the
  dRTCv2 parity is bit-exact (0.0) and a QR/SVD solver would change stored
  controls at the 1e-15 level.
- **L2 — Direct and legacy uniform-left are different procedures.**
  - **Legacy** (`uniform_left_extended_double_fit_v1`) fits a double-clamped
    spline over `chunk + degree·span` actions: 56 for Piper, including 6
    *future* actions. It keeps the first `num_basis` controls, which reproduce
    that extended fit exactly on `[0, chunk]` (test, ≤ 1e-13).
  - **Direct** (`uniform_left_direct_fit_v1`) fits all controls on the
    left-clamped/right-open basis from exactly `chunk` actions plus one
    endpoint padding row.

  Both share the decoding knots. Their controls, calibration, tokens, and
  fingerprints differ, and changing the future actions changes legacy controls
  but cannot affect direct ones (tested). Models trained on one representation
  are not valid for the other.
- **L3 — Last duration.** In a `k`-pair prefix the `k`-th duration only shapes
  right-side support. Tokens for it are carried but cannot influence the
  committed interval. Policies that predict durations should be evaluated with
  that in mind.
- **L4 — Short or unpadded adaptive episodes.** Episodes of ≤ `degree` samples
  yield no records, and `end_padding=False` yields missing tail records (47 of
  60 in a probe). `iter_encoded_episodes` and dataset export raise
  `RuntimeError` instead of writing misaligned data.
- **L5 — Adaptive fitting is episode-global and non-causal.** The
  repeated-tail refit is a least-squares refit that changes the last original
  spans. `fit_tolerance` is not a hard per-record bound: the figure run shows
  max fitting error 9.1e-4 at `fit_tolerance = 2e-3`, but this is not
  guaranteed.
- **L6 — Dataset alignment assumes contiguous `frame_index` at the declared
  fps.** Frame gaps are not detected, and parquet timestamps are loaded but
  unused.
- **L7 — Quantization and fingerprint edge cases:**
  - a constant calibrated channel emits nonzero tokens for out-of-range values
    (for example `[0, 128, 255]`), although they decode to the constant;
  - `tokenizer_id` excludes calibration bounds;
  - `frequency_hz=10` and `10.0` give different fingerprints.

  All are preserved for token and fingerprint compatibility.
- **L8 — Inclusive left-child resampling** (see F2): native samples are
  correct, but `decode(samples=M)` of the left child covers `[0, (s-1)Δt]`.
- **L9 — Insertion multiplicity.** `insert_knot_once` refuses multiplicity
  ≥ degree, by design. Interior knots of multiplicity > `degree` (C⁻¹) are not
  supported by `subdivide`.
- **L10 — Existing plot API.** `plot_reconstruction` does not distinguish the
  executable and right-support regions; the new script does. Its signature was
  left unchanged.
- **L11 — Single environment.** Tests ran on one environment (NumPy 1.26.4,
  SciPy 1.15.3). No other local environment had SciPy ≥ 1.15 with a different
  NumPy, and the package has no CI matrix. NumPy 2.x is unverified.
- **L12 — Not reviewed:** stochastic or real-data accuracy beyond the Piper
  parity set; FITPACK candidate-selection quality; behavior on irregular
  timestamps in the encoder path (only `fit_bspline` honors arbitrary
  timestamps).

## 8. Visual verification

Command (deterministic, CPU only, about 3 s; reruns produce byte-identical
PNGs in this environment):

```bash
python scripts/visual_verification.py --output-dir docs/figures
```

The run asserts 31 numerical checks (all pass) and writes them to
[`figures/visual_verification_metrics.json`](figures/visual_verification_metrics.json).
Data: a seeded 90-step planar figure-eight with a gripper ramp at 20 Hz,
observation 18. All layouts use degree 3, 11 controls, and 2-step spans.

| Figure | Content | What was checked by eye |
|---|---|---|
| [01_reconstruction_and_fitting_error.png](figures/01_reconstruction_and_fitting_error.png) | Target versus continuous spline plus fitting error for `uniform_double`, direct `uniform_left`, legacy extended `uniform_left`, and `adaptive_left`. Knots, the executable interval, and the hatched right-support knots are marked; time in seconds, actions in action units. | Fitting error ≤ 3.2e-4 (uniform) and 9.1e-4 (adaptive), largest near the right end where padding and support dominate. Splines are not drawn beyond the executable end (asserted NaN there). |
| [01b_plot_reconstruction_uniform_left.png](figures/01b_plot_reconstruction_uniform_left.png) | Unchanged packaged `save_reconstruction_plot` output | Renders as before. The gripper dimension at this observation is about 2e-4 noise, so the regularized fit visibly tracks noise. |
| [02_trajectory_2d.png](figures/02_trajectory_2d.png) | 2-D (x, y) trajectory with direct uniform-left (16 steps) and adaptive (35 steps) executable curves, control polygons, and points at distinct knots | Both curves start at the observation and follow the path; the control polygon is not interpolated. |
| [03_subdivision_transfer_boundary_zoom.png](figures/03_subdivision_transfer_boundary_zoom.png) | Adaptive record transferred at a between-knot cut (τ = 0.55 s; 10 pairs → 7 spans) and an existing-knot cut (τ = 0.80 s; 9 pairs → 6 spans). Shows the committed-interval band, a ±2.5-step boundary zoom, and transfer error on its own log axis. | Overlay is indistinguishable; transfer error ≤ 4.4e-16, with exact zeros past the insertion support; the committed end is preserved. |
| [04_position_velocity_acceleration.png](figures/04_position_velocity_acceleration.png) | Position, velocity, and acceleration of the source and both subdivision pieces at a between-knot cut and at a simple-knot cut | One-sided gaps ≤ 1.0e-13 (acceleration); pieces continuous through C² at the knot. |
| [05_quantization_error.png](figures/05_quantization_error.png) | Fitting versus quantization versus requantization-after-transfer error, transfer error on a separate scale, and a vocabulary sweep with the half-bin bound | At vocab 256: quantization 1.27e-3 ≤ half-bin 1.96e-3; requantization ≤ half-bin with 0 clipped controls; measured error tracks the half-bin bound from vocab 16 to 1024. |

The figures supplement the assertions in the test suite and the metrics JSON;
they do not replace them.

## 9. Changed files

Source:
- `src/spline_encoder/core/subdivision.py` (was `subdivision.py`; see §11): F1, F2, F3, F7. Adds `_boehm_insert`
  and `_clamped_piece`; `insert_knot_once` delegates to `_boehm_insert` with
  unchanged interior arithmetic.
- `src/spline_encoder/encoding/encoder.py` (was `encoder.py`): F4 guards.
- `src/spline_encoder/tools/verification.py` (was `verification.py`): F5.

Examples and scripts:
- `examples/encode_chunk.py`: F6.
- `scripts/visual_verification.py` (new): deterministic figures and checks.
- `scripts/compare_output_snapshots.py` (new): bit-exact compatibility check.

Tests (new):
- `tests/test_reference_geometry.py`
- `tests/test_prefix_transfer.py`
- `tests/test_review_regressions.py`

Documentation:
- `README.md`: verification and visualization commands, updated evidence, and
  documented domains, in both the English and Chinese sections.
- `docs/implementation_review.md` (this file).
- `docs/figures/*` (generated assets).

## 10. Suggested focus for the second reviewer

1. Re-derive the boundary branch of `_boehm_insert`. At the right end of a
   right-open vector it uses span `n-1` and counts multiplicity only within
   `knots[:span+1]`; check it against the general Boehm formula.
2. Decide whether L8 (left-child `inclusive_domain`) should become a
   correction. It would change only resampled decodes.
3. Decide whether L6 frame-gap detection should be enforced. It would make
   datasets with dropped frames fail loudly.
4. Re-run the suite and the parity check on NumPy 2.x and a second platform
   (L11).

## 11. Later package reorganization

After this review, the modules were regrouped without numerical edits.

| Reviewed path (`5aede68`) | Current path |
|---|---|
| `fitting.py`, `adaptive.py`, `subdivision.py` | `spline_encoder/core/` |
| `config.py`, `encoder.py`, `result.py`, `quantization.py`, `serialization.py` | `spline_encoder/encoding/` |
| `verification.py`, `visualization.py` | `spline_encoder/tools/` |
| `lerobot.py`, `cli.py` | sibling package `data/splines/` |

All reviewed import paths (`spline_encoder.subdivision`, `spline_encoder.encoder`,
…) still resolve to the same module objects. The `spline_encoder.__all__`
names, signatures, defaults, and `spline-encode` behavior are unchanged.

Evidence after the move:
- the 546-array encoder snapshot is bit-identical to `5aede68` and to the
  pre-reorganization tree;
- the suite is 382 passed, 2 skipped;
- the same suite against the installed wheel in a clean NumPy 2.2.6 venv is 383
  passed, 2 skipped (382 plus an origin check).

That run partly addresses limitation L11 (NumPy 2.x). The final tree,
responsibilities, full mapping, and wheel/CLI checks are in
`docs/preprocessing_integration.md` §10 and the README "Repository layout".

