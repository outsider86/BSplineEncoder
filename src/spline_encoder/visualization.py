"""Optional plotting helpers for visually inspecting spline reconstructions."""

from __future__ import annotations

from pathlib import Path
from numbers import Integral
from typing import Any, Iterable

import numpy as np

from .result import SplineParameters


def _dimensions(values: Iterable[int] | None, action_dim: int) -> tuple[int, ...]:
    raw = tuple(range(action_dim)) if values is None else tuple(values)
    if any(
        not isinstance(index, Integral) or isinstance(index, (bool, np.bool_))
        for index in raw
    ):
        raise ValueError("action dimensions must be integers")
    selected = tuple(int(index) for index in raw)
    if not selected:
        raise ValueError("at least one action dimension must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("action dimensions must be unique")
    if any(index < 0 or index >= action_dim for index in selected):
        raise ValueError("an action dimension is out of range")
    return selected


def plot_reconstruction(
    parameters: SplineParameters,
    reference: Any | None = None,
    *,
    dimensions: Iterable[int] | None = None,
    dense_samples: int = 400,
    show_control_points: bool = True,
    show_knots: bool = True,
    show_dequantized: bool = True,
):
    """Plot target actions, the continuous curve, and its spline geometry.

    Matplotlib is imported lazily so the core encoder remains usable without
    visualization dependencies. The returned ``(figure, axes)`` can be
    further styled or inspected by tests before saving.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "visualization requires `pip install spline-encoder[visualization]`"
        ) from exc

    if dense_samples < 2:
        raise ValueError("dense_samples must be at least 2")
    selected = _dimensions(dimensions, parameters.action_dim)
    target = None
    if reference is not None:
        target = np.asarray(reference, dtype=np.float64)
        expected = (parameters.executable_steps, parameters.action_dim)
        if target.shape != expected or not np.all(np.isfinite(target)):
            raise ValueError(f"reference must be a finite array with shape {expected}")

    columns = min(2, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(7.0 * columns, 3.0 * rows),
        squeeze=False,
        sharex=True,
    )
    axes = tuple(grid.reshape(-1))
    native_times = np.arange(parameters.executable_steps) * parameters.sample_period
    inclusive = bool(parameters.metadata.get("inclusive_domain", False))
    display_end = (
        (parameters.executable_steps - 1) * parameters.sample_period
        if inclusive
        else parameters.executable_steps * parameters.sample_period
    )
    dense_times = np.linspace(0.0, display_end, dense_samples, endpoint=True)
    dense = np.asarray(parameters.scipy_spline()(dense_times), dtype=np.float64)
    dequantized = (
        np.asarray(parameters.scipy_spline(dequantized=True)(dense_times), dtype=np.float64)
        if show_dequantized and parameters.dequantized_control_points is not None
        else None
    )
    knots = np.unique(parameters.knots)
    visible_knots = knots[(knots >= 0.0) & (knots <= display_end)]
    greville = np.asarray([
        parameters.knots[index + 1 : index + parameters.degree + 1].mean()
        for index in range(parameters.num_basis)
    ])
    visible_controls = (greville >= 0.0) & (greville <= display_end)

    for axis, dimension in zip(axes, selected):
        axis.plot(dense_times, dense[:, dimension], label="continuous spline", linewidth=2)
        if dequantized is not None:
            axis.plot(
                dense_times,
                dequantized[:, dimension],
                label="dequantized spline",
                linestyle="--",
                linewidth=1.5,
            )
        if target is not None:
            axis.plot(
                native_times,
                target[:, dimension],
                label="target actions",
                marker="o",
                markersize=3,
                linewidth=1,
            )
        if show_control_points:
            axis.scatter(
                greville[visible_controls],
                parameters.control_points[visible_controls, dimension],
                label="control coefficients",
                marker="x",
                s=28,
                zorder=3,
            )
        if show_knots:
            for knot_index, knot in enumerate(visible_knots):
                axis.axvline(
                    knot,
                    color="0.75",
                    linewidth=0.8,
                    linestyle=":",
                    label="knots" if knot_index == 0 else None,
                    zorder=0,
                )
        axis.set_title(f"Action dimension {dimension}")
        axis.set_ylabel("action")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize="small")

    for axis in axes[len(selected) :]:
        axis.set_visible(False)
    for axis in axes[: len(selected)]:
        axis.set_xlabel("time (seconds)")
    figure.suptitle(
        f"{parameters.metadata.get('mode', 'B-spline')} reconstruction "
        f"({parameters.executable_steps} executable steps)"
    )
    figure.tight_layout()
    return figure, axes[: len(selected)]


def save_reconstruction_plot(
    parameters: SplineParameters,
    path: str | Path,
    reference: Any | None = None,
    **plot_options: Any,
) -> Path:
    """Render :func:`plot_reconstruction` to a file and close its figure."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_reconstruction(parameters, reference, **plot_options)
    import matplotlib.pyplot as plt

    try:
        figure.savefig(target, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return target


__all__ = ["plot_reconstruction", "save_reconstruction_plot"]
