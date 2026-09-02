"""SplineEncoder: continuous-first action B-splines for robotics datasets."""

from .config import (
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
from .encoder import (
    AdaptiveBSplineEncoder,
    AdaptiveLeftBSplineEncoder,
    BaseSplineEncoder,
    UniformBSplineEncoder,
    UniformDoubleBSplineEncoder,
    UniformLeftBSplineEncoder,
    create_encoder,
    encode_action_chunk,
)
from .lerobot import (
    ActionAlignment,
    DatasetEncodingResult,
    EncodedEpisode,
    LeRobotEpisode,
    calibrate_from_lerobot,
    encode_lerobot_dataset,
    infer_action_bounds,
    iter_encoded_episodes,
    load_lerobot_episodes,
)
from .result import SplineGeometry, SplineParameters
from .serialization import (
    ENCODER_STATE_SCHEMA_VERSION,
    encoder_from_state,
    encoder_to_state,
    load_encoder,
    load_tokenizer,
    save_encoder,
    save_tokenizer,
)
from .subdivision import (
    extract_interval,
    extract_left_clamped_prefix,
    insert_knot,
    subdivide,
    subdivide_parameters,
)

__version__ = "0.1.0"

__all__ = [
    "ADAPTIVE_MODES", "ENCODER_MODES", "ENCODER_STATE_SCHEMA_VERSION",
    "UNIFORM_MODES", "ActionAlignment", "AdaptiveBSplineConfig",
    "AdaptiveBSplineEncoder", "AdaptiveLeftBSplineConfig",
    "AdaptiveLeftBSplineEncoder", "BaseSplineEncoder", "DatasetEncodingResult",
    "EncodedEpisode", "LeRobotEpisode", "SplineGeometry", "SplineParameters",
    "UniformBSplineConfig", "UniformBSplineEncoder", "UniformDoubleBSplineConfig",
    "UniformDoubleBSplineEncoder", "UniformLeftBSplineConfig",
    "UniformLeftBSplineEncoder", "benchmark_preset", "calibrate_from_lerobot",
    "config_from_dict", "create_encoder", "encode_action_chunk",
    "encode_lerobot_dataset", "encoder_from_state", "encoder_to_state",
    "extract_interval", "extract_left_clamped_prefix", "infer_action_bounds",
    "insert_knot", "is_adaptive_mode", "is_uniform_mode",
    "iter_encoded_episodes", "load_encoder", "load_lerobot_episodes",
    "load_tokenizer", "preset", "save_encoder", "save_tokenizer", "subdivide",
    "subdivide_parameters",
]
