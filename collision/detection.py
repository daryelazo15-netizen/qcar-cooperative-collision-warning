"""One world-frame detection box (SI units, world coordinates)."""
from dataclasses import dataclass
import math

from .geometry import Footprint


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    yaw: float
    length: float
    width: float
    confidence: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Detection fields must be finite")
        Footprint(self.length, self.width)
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be in [0, 1]")
