"""Compatibility alias for :mod:`data.splines.lerobot` (LeRobot action I/O and sidecar export).

The canonical implementation moved to the sibling ``data`` package; this
module path keeps resolving to the same module object.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("data.splines.lerobot")
