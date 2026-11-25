"""fastwan-litert: FastWan 2.1 1.3B on LiteRT.

The model is cut into `.tflite` graphs (`docs/graphs.md`); this package
exports them from a Diffusers checkpoint (`export`), runs them on the desktop
through `ai-edge-litert` (`runtime`, `pipeline`), and carries the host math
every other runtime ports (`config`, `geometry`, `schedule`, `rope`, `rng`,
`host`).
"""

from .config import TaeConfig, WanConfig, WAN21_LATENT_MEAN, WAN21_LATENT_STD
from .geometry import Geometry
from .schedule import DmdSchedule
from .rng import Rng

__all__ = [
    "TaeConfig",
    "WanConfig",
    "WAN21_LATENT_MEAN",
    "WAN21_LATENT_STD",
    "Geometry",
    "DmdSchedule",
    "Rng",
]
__version__ = "0.1.0"
