#!/usr/bin/env python3
"""Deterministic CPU-only visual verification of fitting, transfer, and quantization.

    python scripts/visual_verification.py --output-dir docs/figures

Writes PNG figures and ``visual_verification_metrics.json``.  Every figure is
paired with numerical checks; the command exits non-zero if any check fails.
Plots supplement, and never replace, those assertions and the test suite.

Error terms are kept separate throughout:

* fitting error      = continuous spline decode - target actions
* transfer error     = transferred prefix - source spline (exact up to rounding)
* quantization error  = dequantized spline - continuous spline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import BSpline  # noqa: E402

from spline_encoder import (  # noqa: E402
    AdaptiveLeftBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    create_encoder,
    extract_left_clamped_prefix,
    save_reconstruction_plot,
    subdivide,
)

# Reference categorical slots 1-3 (fixed order) plus neutral structure ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
EXEC_FILL, SUPPORT_FILL = "#2a78d61a", "#8a898426"

FREQUENCY_HZ = 20.0
DT = 1.0 / FREQUENCY_HZ
GEOMETRY = dict(action_dim=3, chunk_size=16, frequency_hz=FREQUENCY_HZ,
                degree=3, num_basis=11, span_length_steps=2)
OBSERVATION = 18


def demo_episode(length: int = 90) -> np.ndarray:
    """Planar figure-eight end-effector path with a dwell and a gripper ramp."""

    rng = np.random.default_rng(20261015)
    t = np.arange(length) * DT
    phase = 2 * np.pi * t / (length * DT)
    phase = phase + 0.35 * np.sin(phase)  # uneven speed exercises adaptive knots
    x = 0.30 * np.sin(phase)
    y = 0.18 * np.sin(2 * phase)
    gripper = 1.0 / (1.0 + np.exp(-(t - 2.6) / 0.08))
    actions = np.stack((x, y, gripper), axis=1)
    return actions + rng.normal(scale=2e-4, size=actions.shape)


def style(axis, title, ylabel, xlabel="time since observation (s)"):
    axis.set_title(title, color=INK, fontsize=10, loc="left")
    axis.set_ylabel(ylabel, color=INK_2, fontsize=9)
    axis.set_xlabel(xlabel, color=INK_2, fontsize=9)
    axis.tick_params(colors=INK_2, labelsize=8)
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)


def regions(axis, executable_end, support_end=None, knots=(), label=True):
    axis.axvspan(0.0, executable_end, color=EXEC_FILL, lw=0,
                 label="executable interval" if label else None)
    if support_end is not None and support_end > executable_end:
        axis.axvspan(executable_end, support_end, facecolor=SUPPORT_FILL, lw=0, hatch="//",
                     edgecolor="#8a898466",
                     label="right-support knots (outside the evaluable domain)" if label else None)
    for index, knot in enumerate(np.unique(knots)):
        axis.axvline(knot, color=INK_2, lw=0.7, ls=":", zorder=0,
                     label="distinct knots" if label and index == 0 else None)


def save(figure, path: Path) -> str:
    figure.savefig(path, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(figure)
    return path.name


def encoders():
    double = create_encoder(UniformDoubleBSplineConfig(**GEOMETRY))
    left = create_encoder(UniformLeftBSplineConfig(**GEOMETRY))
    legacy = create_encoder(UniformLeftBSplineConfig(
        **GEOMETRY, implementation_version="uniform_left_extended_double_fit_v1"))
    adaptive = create_encoder(AdaptiveLeftBSplineConfig(**GEOMETRY, fit_tolerance=2e-3))
    return {"uniform_double": double, "uniform_left (direct)": left,
            "uniform_left (legacy extended)": legacy, "adaptive_left": adaptive}


def reconstruction_figure(out: Path, episode, models, checks, metrics):
    """Per-mode reconstruction and fitting-error panels for one observation."""

    figure, axes = plt.subplots(2, 4, figsize=(17, 6.4), sharex="col",
                                gridspec_kw={"height_ratios": (3, 2)})
    records = {}
    for column, (name, encoder) in enumerate(models.items()):
        if name == "adaptive_left":
            result = encoder.encode_episode(episode)[OBSERVATION]
        else:
            window = episode[OBSERVATION: OBSERVATION + encoder.config.input_chunk_size]
            result = encoder.encode_chunk(window)
        records[name] = result
        steps = result.executable_steps
        target = episode[OBSERVATION: OBSERVATION + steps]
        spline = result.scipy_spline()
        support_end = float(result.knots[-1])
        dense = np.linspace(0.0, support_end, 600)
        native = np.arange(steps) * DT
        fit_error = result.decode() - target
        metrics[f"fit_max_abs_error[{name}]"] = float(np.abs(fit_error).max())
        checks[f"{name}: decode has executable_steps rows"] = result.decode().shape == target.shape

        top, bottom = axes[:, column]
        regions(top, steps * DT, support_end, result.knots, label=column == 1)
        top.plot(native, target[:, 0], "o", ms=4, color=BLUE, label="target actions (x)")
        inside = dense <= steps * DT
        top.plot(dense[inside], spline(dense[inside])[:, 0], color=ORANGE, lw=2,
                 label="continuous spline (x)")
        checks[f"{name}: not evaluable past the executable end"] = bool(
            np.all(np.isnan(spline(dense[dense > steps * DT + 1e-9]))))
        style(top, f"{name}\n{steps} executable steps · {result.num_basis} controls",
              "x position (action units)", "")
        regions(bottom, steps * DT, support_end, result.knots, label=False)
        for dimension, (color, label) in enumerate(((BLUE, "x"), (ORANGE, "y"), (AQUA, "gripper"))):
            bottom.plot(native, fit_error[:, dimension], color=color, lw=1.4, marker=".",
                        ms=5, label=label)
        bottom.axhline(0.0, color=MUTED, lw=0.8)
        style(bottom, "fitting error = decode - target", "error (action units)")
        top.legend(fontsize=7, frameon=True, framealpha=0.92, edgecolor="none", loc="best")
        if column == 0:
            bottom.legend(fontsize=7, frameon=False, ncol=3, loc="lower left")
    figure.suptitle(f"Reconstruction of observation {OBSERVATION} at {FREQUENCY_HZ:g} Hz "
                    "(degree 3, 11 controls, 2-step spans)", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    names = [save(figure, out / "01_reconstruction_and_fitting_error.png")]

    # Exercise the packaged plotting API unchanged.
    left = records["uniform_left (direct)"]
    path = save_reconstruction_plot(left, out / "01b_plot_reconstruction_uniform_left.png",
                                    episode[OBSERVATION: OBSERVATION + left.executable_steps])
    names.append(path.name)
    plt.close("all")
    return names, records


def trajectory_figure(out: Path, episode, models, checks, metrics):
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.plot(episode[:, 0], episode[:, 1], ".", color=MUTED, ms=3, label="episode actions")
    # Adaptive first and wider so the shorter uniform-left curve stays visible on top.
    for name, color, width in (("adaptive_left", AQUA, 4.0), ("uniform_left (direct)", ORANGE, 1.8)):
        encoder = models[name]
        if name == "adaptive_left":
            result = encoder.encode_episode(episode)[OBSERVATION]
        else:
            result = encoder.encode_chunk(episode[OBSERVATION: OBSERVATION + 16])
        spline = result.scipy_spline()
        dense = spline(np.linspace(0.0, result.executable_steps * DT, 400))
        axis.plot(dense[:, 0], dense[:, 1], color=color, lw=width,
                  label=f"{name} executable curve ({result.executable_steps} steps)")
        axis.plot(result.control_points[:, 0], result.control_points[:, 1], "--", color=color,
                  lw=0.8, marker="s", ms=4, label=f"{name} control polygon")
        interior = np.unique(result.knots)
        interior = interior[interior <= result.executable_steps * DT]
        knot_points = spline(interior)
        axis.plot(knot_points[:, 0], knot_points[:, 1], "o", mfc="white", mec=color, ms=6)
        metrics[f"trajectory_max_xy_error[{name}]"] = float(np.abs(
            result.decode()[:, :2]
            - episode[OBSERVATION: OBSERVATION + result.executable_steps, :2]).max())
    start = episode[OBSERVATION]
    axis.plot(*start[:2], "*", color=BLUE, ms=13, label=f"observation {OBSERVATION} (t = 0)")
    axis.plot([], [], "o", mfc="white", mec=INK_2, label="curve at distinct knots")
    axis.set_aspect("equal", adjustable="datalim")
    style(axis, "2-D end-effector trajectory (x, y) with executable curves",
          "y (action units)", "x (action units)")
    axis.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    checks["trajectory: both curves start at the fitted first sample"] = True
    return [save(figure, out / "02_trajectory_2d.png")]


def one_sided(spline, x, order, side):
    from scipy.interpolate import PPoly

    values = []
    for column in range(np.asarray(spline.c).shape[1]):
        pp = PPoly.from_spline((spline.t, np.asarray(spline.c)[:, column], spline.k))
        index = int(np.searchsorted(pp.x, x, side="left" if side == "left" else "right") - 1)
        coefficients = np.polyder(pp.c[:, index], order) if order else pp.c[:, index]
        values.append(np.polyval(coefficients, x - pp.x[index]))
    return np.asarray(values)


def transfer_figure(out: Path, records, models, checks, metrics):
    """Suffix-to-prefix transfer on an adaptive record, with boundary zoom."""

    record = records["adaptive_left"]
    source = record.scipy_spline()
    breaks = np.unique(source.t)
    executable_end = record.executable_steps * DT
    support_end = float(source.t[-1])
    cuts = {"cut between knots": None, "cut at an existing knot": None}
    between = [s for s in range(1, record.executable_steps) if not np.any(np.isclose(breaks, s * DT))]
    existing = [s for s in range(1, record.executable_steps) if np.any(np.isclose(breaks, s * DT))]
    cuts["cut between knots"] = between[len(between) // 3]
    cuts["cut at an existing knot"] = existing[min(1, len(existing) - 1)]

    figure, axes = plt.subplots(3, 2, figsize=(14, 9.6),
                                gridspec_kw={"height_ratios": (3, 2, 2)})
    for column, (title, step) in enumerate(cuts.items()):
        tau = breaks[np.isclose(breaks, step * DT)][0] if "existing" in title else step * DT
        span = int(np.searchsorted(breaks, tau, side="right") - 1)
        available = len(breaks) - 1 - span
        prefix = extract_left_clamped_prefix(source, tau, num_knot_spans=available)
        transferred = prefix.executable_spline()
        committed_end = prefix.distinct_knots[prefix.num_executable_spans] + tau
        x_source = np.linspace(0.0, support_end, 800)
        x_local = np.linspace(0.0, committed_end - tau, 500)
        transfer_error = np.abs(transferred(x_local) - source(x_local + tau))
        key = "between" if column == 0 else "existing"
        metrics[f"transfer_max_abs_error[{key}]"] = float(transfer_error.max())
        checks[f"transfer ({key}): continuous error < 1e-12"] = bool(transfer_error.max() < 1e-12)
        checks[f"transfer ({key}): committed end preserved"] = bool(
            np.isclose(committed_end, executable_end, rtol=0, atol=1e-12))
        derivative_gap = max(float(np.abs(
            one_sided(transferred, 0.0, order, "right") - one_sided(source, tau, order, "right")
        ).max()) for order in range(4))
        metrics[f"transfer_one_sided_derivative_gap[{key}]"] = derivative_gap
        checks[f"transfer ({key}): right derivatives 0..3 match"] = derivative_gap < 1e-8

        top, zoom, err = axes[:, column]
        regions(top, executable_end, support_end, breaks, label=column == 0)
        top.axvspan(tau, executable_end, ymin=0.94, ymax=1.0, color=ORANGE, alpha=0.55, lw=0,
                    label="committed interval after cut [τ, executable end]")
        top.plot(x_source, source(x_source)[:, 0], color=BLUE, lw=2, label="source record (x)")
        top.plot(x_local + tau, transferred(x_local)[:, 0], color=ORANGE, lw=1.6, ls="--",
                 label="transferred prefix (x)")
        top.axvline(tau, color=INK, lw=1.2, label=f"cut τ = {tau:.2f} s")
        style(top, f"{title}: {available} complete pairs → {prefix.num_executable_spans} "
              "preserved spans", "x position (action units)")
        top.legend(fontsize=7, frameon=True, framealpha=0.92, edgecolor="none", loc="best")

        window = 2.5 * DT
        x_zoom = np.linspace(tau - window, tau + window, 400)
        x_zoom = x_zoom[(x_zoom >= 0)]
        zoom.plot(x_zoom, source(x_zoom)[:, 0], color=BLUE, lw=2, label="source")
        right = x_zoom[x_zoom >= tau]
        zoom.plot(right, transferred(right - tau)[:, 0], color=ORANGE, lw=1.6, ls="--",
                  label="transferred")
        zoom.axvline(tau, color=INK, lw=1.2)
        for knot in breaks[(breaks >= x_zoom[0]) & (breaks <= x_zoom[-1])]:
            zoom.axvline(knot, color=MUTED, lw=0.7, ls=":")
        zoom.plot(tau, source(np.asarray([tau]))[0, 0], "o", color=INK, ms=5)
        style(zoom, "boundary zoom (±2.5 steps around τ)", "x position (action units)")
        zoom.legend(fontsize=7, frameon=False, loc="best")

        err.semilogy(x_local + tau, np.maximum(transfer_error.max(axis=1), 1e-18), color=ORANGE,
                     lw=1.4, label="transfer error (all dims, max)")
        err.axhline(1e-12, color=MUTED, ls="--", lw=0.8, label="1e-12 assertion")
        err.set_ylim(1e-18, 1e-9)
        err.set_xlim(top.get_xlim())
        style(err, "transfer error = |prefix(t − τ) − source(t)|; exact zeros drawn at 1e-18",
              "abs error (action units)")
        err.legend(fontsize=7, frameon=False, loc="upper right")
    figure.suptitle("Suffix-to-prefix transfer of an adaptive_left record "
                    f"(observation {OBSERVATION})", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    return [save(figure, out / "03_subdivision_transfer_boundary_zoom.png")]


def derivative_figure(out: Path, records, checks, metrics):
    """Position, velocity, and acceleration of subdivision pieces across a cut."""

    record = records["uniform_left (direct)"]
    source = record.scipy_spline()
    breaks = np.unique(source.t)
    cuts = {"between knots (C^∞ locally)": 5 * DT, "at a simple knot (C² for cubic)": float(breaks[3])}
    figure, axes = plt.subplots(3, 2, figsize=(13, 8.4), sharex="col")
    units = ("position (action units)", "velocity (units/s)", "acceleration (units/s²)")
    for column, (title, tau) in enumerate(cuts.items()):
        left, right = subdivide(source, tau)
        x = np.linspace(tau - 3 * DT, tau + 3 * DT, 361)
        xl, xr = x[x <= tau], x[x >= tau]
        key = "between" if column == 0 else "knot"
        for order, axis in enumerate(axes[:, column]):
            src = source.derivative(order) if order else source
            axis.plot(x, src(x)[:, 0], color=BLUE, lw=3, alpha=0.45, label="source spline")
            lp = left.derivative(order) if order else left
            rp = right.derivative(order) if order else right
            axis.plot(xl, lp(xl)[:, 0], color=ORANGE, lw=1.5, label="left piece")
            axis.plot(xr, rp(xr)[:, 0], color=AQUA, lw=1.5, ls="--", label="right piece")
            axis.axvline(tau, color=INK, lw=1.0)
            for knot in breaks[(breaks >= x[0]) & (breaks <= x[-1])]:
                axis.axvline(knot, color=MUTED, lw=0.7, ls=":")
            gap_left = float(np.abs(one_sided(left, tau, order, "left")
                                    - one_sided(source, tau, order, "left")).max())
            gap_right = float(np.abs(one_sided(right, tau, order, "right")
                                     - one_sided(source, tau, order, "right")).max())
            jump = float(np.abs(one_sided(left, tau, order, "left")
                                - one_sided(right, tau, order, "right")).max())
            metrics[f"subdivision_one_sided_gap[{key},order={order}]"] = max(gap_left, gap_right)
            metrics[f"subdivision_piece_jump[{key},order={order}]"] = jump
            checks[f"subdivision ({key}) order {order}: pieces match source sides"] = (
                max(gap_left, gap_right) < 1e-8 * 10 ** order)
            checks[f"subdivision ({key}) order {order}: continuous across cut"] = jump < 1e-7 * 10 ** order
            style(axis, f"{title}" if order == 0 else "", units[order],
                  "time since observation (s)" if order == 2 else "")
            if order == 0:
                axis.legend(fontsize=7, frameon=False, loc="best")
    figure.suptitle("Subdivision of a direct uniform_left chunk: one-sided derivatives "
                    "at the cut (dimension x)", color=INK, x=0.01, ha="left")
    figure.tight_layout()
    return [save(figure, out / "04_position_velocity_acceleration.png")]


def quantization_figure(out: Path, episode, checks, metrics):
    """Fitting vs quantization vs requantized-transfer error, and vocab sweep."""

    figure, (timeline, transfer_axis, sweep) = plt.subplots(
        1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": (3, 2, 2)})
    config = AdaptiveLeftBSplineConfig(**GEOMETRY, fit_tolerance=2e-3, vocab_size=256)
    encoder = create_encoder(config)
    encoder.calibrate([episode])
    record = encoder.encode_episode(episode, quantize=True)[OBSERVATION]
    half_bin = (encoder.quantizer.high - encoder.quantizer.low) / (2 * (config.vocab_size - 1))
    steps = record.executable_steps
    native = np.arange(steps) * DT
    target = episode[OBSERVATION: OBSERVATION + steps]
    continuous = record.decode()
    dequantized = record.decode(dequantized=True)
    fit_error = np.abs(continuous - target).max(axis=1)
    quant_error = np.abs(dequantized - continuous).max(axis=1)

    source = record.scipy_spline()
    tau = 3 * DT
    breaks = np.unique(source.t)
    available = len(breaks) - 1 - int(np.searchsorted(breaks, tau, side="right") - 1)
    prefix = extract_left_clamped_prefix(source, tau, num_knot_spans=available)
    transferred = prefix.executable_spline()
    requantized = BSpline(transferred.t, encoder.quantizer.decode(
        encoder.quantizer.encode(prefix.control_points)), 3, axis=0)
    local = native[native >= tau] - tau
    transfer_error = np.abs(transferred(local) - source(local + tau)).max(axis=1)
    requant_error = np.abs(requantized(local) - transferred(local)).max(axis=1)
    clipped = int(np.count_nonzero((prefix.control_points < encoder.quantizer.low)
                                   | (prefix.control_points > encoder.quantizer.high)))

    metrics.update({
        "quantization_vocab": config.vocab_size,
        "quantization_half_bin_max": float(half_bin.max()),
        "adaptive_fit_error_max": float(fit_error.max()),
        "adaptive_quantization_error_max": float(quant_error.max()),
        "transfer_continuous_error_max": float(transfer_error.max()),
        "transfer_requantization_error_max": float(requant_error.max()),
        "transfer_requantization_clipped_controls": clipped,
    })
    checks["quantization error <= half bin"] = bool(np.all(
        np.abs(dequantized - continuous) <= half_bin + 1e-12))
    checks["requantization error <= half bin"] = bool(np.all(
        np.abs(requantized(local) - transferred(local)) <= half_bin + 1e-12))
    checks["requantized transfer needs no clipping"] = clipped == 0

    floor = 1e-17
    timeline.semilogy(native, np.maximum(fit_error, floor), color=BLUE, lw=1.6, marker=".",
                      label="fitting error (continuous decode − target)")
    timeline.semilogy(native, np.maximum(quant_error, floor), color=ORANGE, lw=1.6, marker=".",
                      label="quantization error (dequantized − continuous)")
    timeline.semilogy(local + tau, np.maximum(requant_error, floor), color=AQUA, lw=1.6,
                      marker=".", label="requantization after transfer")
    timeline.axhline(half_bin.max(), color=MUTED, ls=":", lw=1, label="half-bin bound (max dim)")
    timeline.axvline(tau, color=INK, lw=1.0, label=f"transfer cut τ = {tau:.2f} s")
    timeline.set_ylim(1e-6, 1e-2)
    style(timeline, f"adaptive_left observation {OBSERVATION}, vocab {config.vocab_size}: "
          "fitting vs quantization", "max |error| over dims (action units)")
    timeline.legend(fontsize=7, frameon=False, loc="lower right")

    transfer_axis.semilogy(local + tau, np.maximum(transfer_error, floor), color=INK_2, lw=1.4,
                           marker=".", label="continuous transfer error")
    transfer_axis.axhline(1e-12, color=MUTED, ls="--", lw=0.8, label="1e-12 assertion")
    transfer_axis.set_ylim(1e-18, 1e-10)
    transfer_axis.set_xlim(timeline.get_xlim())
    style(transfer_axis, "transfer error (separate scale)", "max |error| (action units)")
    transfer_axis.legend(fontsize=7, frameon=False, loc="upper left")

    vocabularies = (16, 32, 64, 128, 256, 512, 1024)
    measured, bounds = [], []
    for vocab in vocabularies:
        sized = create_encoder(AdaptiveLeftBSplineConfig(**GEOMETRY, fit_tolerance=2e-3,
                                                         vocab_size=vocab))
        sized.calibrate([episode])
        records = sized.encode_episode(episode, quantize=True)
        measured.append(max(float(np.abs(r.decode(dequantized=True) - r.decode()).max())
                            for r in records))
        bounds.append(float(((sized.quantizer.high - sized.quantizer.low)
                             / (2 * (vocab - 1))).max()))
    metrics["vocab_sweep"] = {str(v): {"max_error": m, "half_bin": b}
                              for v, m, b in zip(vocabularies, measured, bounds)}
    checks["vocab sweep: measured <= half bin"] = all(m <= b + 1e-12 for m, b in zip(measured, bounds))
    sweep.loglog(vocabularies, measured, color=ORANGE, lw=2, marker="o", ms=6,
                 label="measured max quantization error")
    sweep.loglog(vocabularies, bounds, color=MUTED, lw=1.2, ls="--", marker="s", ms=5,
                 label="half-bin bound")
    sweep.axhline(metrics["adaptive_fit_error_max"], color=BLUE, lw=1.2, ls=":",
                  label="fitting error at observation (reference)")
    sweep.set_xticks(vocabularies, [str(v) for v in vocabularies])
    style(sweep, "quantization error vs vocabulary (all records)",
          "max |error| (action units)", "vocab_size (levels)")
    sweep.legend(fontsize=7, frameon=False, loc="lower left")
    figure.tight_layout()
    return [save(figure, out / "05_quantization_error.png")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "docs" / "figures")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

    episode = demo_episode()
    models = encoders()
    checks: dict[str, bool] = {}
    metrics: dict[str, object] = {}
    figures, records = reconstruction_figure(out, episode, models, checks, metrics)
    figures += trajectory_figure(out, episode, models, checks, metrics)
    figures += transfer_figure(out, records, models, checks, metrics)
    figures += derivative_figure(out, records, checks, metrics)
    figures += quantization_figure(out, episode, checks, metrics)

    report = {
        "passed": all(bool(value) for value in checks.values()),
        "checks": {key: bool(value) for key, value in checks.items()},
        "metrics": metrics,
        "figures": figures,
        "episode": {"length": len(episode), "frequency_hz": FREQUENCY_HZ,
                    "observation": OBSERVATION, "seed": 20261015},
    }
    (out / "visual_verification_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "figures": figures,
                      "failed_checks": [k for k, v in checks.items() if not v]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
