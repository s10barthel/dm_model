"""Geometry for the minimum horizontal covering sector of other outfield players."""

from dataclasses import dataclass
from math import hypot

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Wedge


@dataclass(frozen=True)
class CoveringSector:
    center: tuple[float, float]
    start_deg: float
    width_deg: float

    def rotated(self, field_size: tuple[float, float]) -> "CoveringSector":
        x, y = self.center
        return CoveringSector(
            (field_size[0] - x, field_size[1] - y), (self.start_deg + 180) % 360, self.width_deg,
        )


def covering_sector(center, targets) -> CoveringSector | None:
    """Take the complement of the largest gap in the CCW bearings (first gap on ties)."""
    center = np.asarray(center, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if not targets.size:
        return None
    offsets = targets - center
    if not np.isfinite(offsets).all():
        return None
    if (np.hypot(offsets[:, 0], offsets[:, 1]) <= 1e-9).any():
        return None  # A coincident point has no bearing.
    bearings = np.sort(np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0])) % 360)
    gaps = np.diff(np.r_[bearings, bearings[0] + 360])
    gap = int(np.argmax(gaps))
    return CoveringSector(tuple(center), float(bearings[(gap + 1) % len(bearings)]), float(360 - gaps[gap]))


def draw_covering_sector(ax, sector: CoveringSector, field_size, *, pitch_control=False) -> Wedge | None:
    if sector.width_deg <= 0:
        return None  # A zero-width sector has no filled area.
    length, width = field_size
    x, y = sector.center
    # The arc is outside the entire rectangle, even if the apex is off the pitch.
    radius = 2 * hypot(max(abs(x), abs(length - x)), max(abs(y), abs(width - y)))
    color = (1.0, 1.0, 1.0, 0.75) if pitch_control else (0.5, 0.5, 0.5, 0.25)
    patch = Wedge(
        sector.center, radius, sector.start_deg, sector.start_deg + sector.width_deg,
        facecolor=color, edgecolor="none", zorder=-0.5,
    )
    ax.add_patch(patch)
    # A data-space path preserves the separate axes clip box (e.g. a zoomed view).
    patch.set_clip_path(Path([(0, 0), (length, 0), (length, width), (0, width), (0, 0)]), ax.transData)
    return patch
