"""SplineEncoder: continuous-first action B-splines for robotics datasets.

Layout: ``core`` (knots, fitting, adaptive fitting, subdivision), ``encoding``
(configs, encoders, results, quantization, serialization) and ``tools``
(verification, visualization).  None of them imports dataset or video
dependencies.  LeRobot dataset I/O lives in the sibling ``data`` package; its
historical names below are re-exported lazily on first access.
"""

import importlib
import sys

from .encoding.config import (
    ADAPTIVE_MODES,
    ENCODER_MODES,
    UNIFORM_MODES,
    AdaptiveBSplineConfig,
    AdaptiveLeftBSplineConfig,
    UniformBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    benchmark_preset,
    config_from_dict,
    is_adaptive_mode,
    is_uniform_mode,
    preset,
)
from .encoding.encoder import (
    AdaptiveBSplineEncoder,
    AdaptiveLeftBSplineEncoder,
    BaseSplineEncoder,
    UniformBSplineEncoder,
    UniformDoubleBSplineEncoder,
    UniformLeftBSplineEncoder,
    create_encoder,
    encode_action_chunk,
)
from .core.fitting import (
    BSplineControlPointFit,
    bspline_design_matrix,
    evaluate_bspline,
    fit_bspline,
    fit_bspline_control_points,
    fit_control_points,
    make_clamped_knots,
    make_uniform_clamped_knots,
    make_uniform_left_clamped_knots,
)
from .encoding.result import SplineGeometry, SplineParameters
from .encoding.serialization import (
    ENCODER_STATE_SCHEMA_VERSION,
    encoder_from_state,
    encoder_to_state,
    load_encoder,
    load_tokenizer,
    save_encoder,
    save_tokenizer,
)
from .core.subdivision import (
    extract_interval,
    extract_left_clamped_prefix,
    insert_knot,
    subdivide,
    subdivide_parameters,
)
from .tools.verification import (
    ReconstructionMetrics,
    reconstruction_metrics,
    verify_reconstruction,
    verify_spline_parameters,
)
from .tools.visualization import plot_reconstruction, save_reconstruction_plot

__version__ = "0.1.0"

# Compatibility: the pre-reorganization module paths (``spline_encoder.fitting``
# etc.) resolve to the SAME module objects as their canonical locations.
_MODULE_ALIASES = {
    "adaptive": "core.adaptive", "fitting": "core.fitting", "subdivision": "core.subdivision",
    "config": "encoding.config", "encoder": "encoding.encoder", "result": "encoding.result",
    "quantization": "encoding.quantization", "serialization": "encoding.serialization",
    "verification": "tools.verification", "visualization": "tools.visualization",
}
for _old, _new in _MODULE_ALIASES.items():
    _module = importlib.import_module(f"{__name__}.{_new}")
    sys.modules[f"{__name__}.{_old}"] = _module
    globals()[_old] = _module
del _old, _new, _module

# LeRobot dataset I/O moved to ``data.splines.lerobot``; resolved on first use
# so importing spline_encoder never imports the data package.
_LEROBOT_NAMES = frozenset({
    "ActionAlignment", "DatasetEncodingResult", "EncodedEpisode", "LeRobotEpisode",
    "calibrate_from_lerobot", "encode_lerobot_dataset", "infer_action_bounds",
    "iter_encoded_episodes", "load_lerobot_episodes",
})


def __getattr__(name: str):
    if name in _LEROBOT_NAMES:
        value = getattr(importlib.import_module("data.splines.lerobot"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ADAPTIVE_MODES", "ENCODER_MODES", "ENCODER_STATE_SCHEMA_VERSION",
    "UNIFORM_MODES", "ActionAlignment", "AdaptiveBSplineConfig",
    "AdaptiveBSplineEncoder", "AdaptiveLeftBSplineConfig",
    "AdaptiveLeftBSplineEncoder", "BaseSplineEncoder", "DatasetEncodingResult",
    "BSplineControlPointFit", "EncodedEpisode", "LeRobotEpisode", "SplineGeometry",
    "SplineParameters",
    "UniformBSplineConfig", "UniformBSplineEncoder", "UniformDoubleBSplineConfig",
    "UniformDoubleBSplineEncoder", "UniformLeftBSplineConfig",
    "UniformLeftBSplineEncoder", "benchmark_preset", "calibrate_from_lerobot",
    "bspline_design_matrix", "config_from_dict", "create_encoder",
    "encode_action_chunk", "evaluate_bspline", "fit_bspline",
    "fit_bspline_control_points", "fit_control_points",
    "encode_lerobot_dataset", "encoder_from_state", "encoder_to_state",
    "extract_interval", "extract_left_clamped_prefix", "infer_action_bounds",
    "insert_knot", "is_adaptive_mode", "is_uniform_mode",
    "iter_encoded_episodes", "load_encoder", "load_lerobot_episodes",
    "load_tokenizer", "make_clamped_knots", "make_uniform_clamped_knots",
    "make_uniform_left_clamped_knots",
    "preset", "save_encoder", "save_tokenizer", "subdivide",
    "subdivide_parameters", "ReconstructionMetrics", "plot_reconstruction",
    "reconstruction_metrics", "save_reconstruction_plot",
    "verify_reconstruction", "verify_spline_parameters",
]
