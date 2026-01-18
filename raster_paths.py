"""
raster_paths.py

Pure path-generation algorithms for rastering / beam steering.

This module MUST NOT:
- import PyQt / pyqtgraph
- talk to motors / DLLs
- depend on UI objects

It should be deterministic and unit-testable.

Coordinate conventions:
- "target space" here is whatever coordinate system your plot/clicks use.
  (It can be pixels, or mm if you apply scaling in the UI.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay

TargetXY = Tuple[float, float]
Bounds = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


# -------------------------
# Helpers
# -------------------------

def bounds_from_points(points: Sequence[TargetXY]) -> Bounds:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys)))


def within_bounds(x: float, y: float, bounds: Optional[Bounds]) -> bool:
    if bounds is None:
        return True
    xmin, xmax, ymin, ymax = bounds
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def arange_inclusive(start: float, stop: float, step: float, *, include_stop: bool = True, eps: float = 1e-12) -> np.ndarray:
    """
    Like np.arange, but optionally includes the stop value when it lands on-grid.
    """
    if step == 0:
        raise ValueError("step must be nonzero")
    n = int(np.floor((stop - start) / step + eps)) + 1
    if n <= 0:
        return np.array([], dtype=float)
    arr = start + step * np.arange(n, dtype=float)
    if include_stop and len(arr) > 0:
        if abs(arr[-1] - stop) <= max(eps, abs(step) * 1e-9):
            arr[-1] = stop
    return arr


def segments_from_points(points: Sequence[TargetXY]) -> List[Tuple[TargetXY, TargetXY]]:
    """
    Returns line segments between successive points (for direction visualization).
    """
    segs: List[Tuple[TargetXY, TargetXY]] = []
    for i in range(len(points) - 1):
        segs.append((points[i], points[i + 1]))
    return segs


def collect_points(it: Iterable[TargetXY], max_points: Optional[int] = None) -> List[TargetXY]:
    pts: List[TargetXY] = []
    for i, p in enumerate(it):
        pts.append((float(p[0]), float(p[1])))
        if max_points is not None and i + 1 >= int(max_points):
            break
    return pts


# -------------------------
# Square raster (serpentine)
# -------------------------

def iter_square_raster_x(bounds: Bounds, xstep: float, ystep: float, *, include_start: bool = True) -> Iterator[TargetXY]:
    """
    Serpentine scan: primary motion along X, then step in Y when reaching edge.

    Starts at (xmin, ymax) like the legacy code.
    """
    xmin, xmax, ymin, ymax = bounds
    if xstep <= 0 or ystep <= 0:
        raise ValueError("xstep and ystep must be > 0")

    # Y rows from top to bottom (ymax -> ymin)
    ys = arange_inclusive(ymax, ymin, -ystep, include_stop=True)
    xs_fwd = arange_inclusive(xmin, xmax, xstep, include_stop=True)
    xs_rev = xs_fwd[::-1]

    first = True
    for row_idx, y in enumerate(ys):
        xs = xs_fwd if (row_idx % 2 == 0) else xs_rev
        for col_idx, x in enumerate(xs):
            if not include_start and first:
                first = False
                continue
            first = False
            yield (float(x), float(y))


def iter_square_raster_y(bounds: Bounds, xstep: float, ystep: float, *, include_start: bool = True) -> Iterator[TargetXY]:
    """
    Serpentine scan: primary motion along Y, then step in X when reaching edge.

    Starts at (xmin, ymax) like the legacy code.
    """
    xmin, xmax, ymin, ymax = bounds
    if xstep <= 0 or ystep <= 0:
        raise ValueError("xstep and ystep must be > 0")

    xs = arange_inclusive(xmin, xmax, xstep, include_stop=True)
    ys_fwd = arange_inclusive(ymax, ymin, -ystep, include_stop=True)
    ys_rev = ys_fwd[::-1]

    first = True
    for col_idx, x in enumerate(xs):
        ys = ys_fwd if (col_idx % 2 == 0) else ys_rev
        for row_idx, y in enumerate(ys):
            if not include_start and first:
                first = False
                continue
            first = False
            yield (float(x), float(y))


# -------------------------
# Spiral raster (inward)
# -------------------------

def iter_spiral_inward(
    origin: TargetXY,
    radius: float,
    step: float,
    angle_step: float,
    angle_step_change: float = 0.0,
    *,
    bounds: Optional[Bounds] = None,
) -> Iterator[TargetXY]:
    """
    Inward spiral around origin. Mirrors the legacy SpiralRaster.preview_path algorithm.

    radius: starting radius
    step: decrement in radius after each 2*pi wrap
    angle_step: delta alpha per point
    angle_step_change: increment to angle_step after each wrap (optional)

    If bounds is provided, points outside bounds are skipped.
    """
    ox, oy = float(origin[0]), float(origin[1])
    r = float(radius)
    st = float(step)
    alpha = 0.0
    d_alpha = float(angle_step)

    if st <= 0 or r < 0:
        raise ValueError("radius must be >= 0 and step must be > 0")
    if d_alpha <= 0:
        raise ValueError("angle_step must be > 0")

    # Include the origin as first point? Legacy started at origin but then immediately moved.
    # We'll yield points on the spiral only (not origin), matching the legacy preview that appended computed points.
    while r >= 0:
        if alpha < 2 * np.pi:
            x = ox + r * np.cos(alpha)
            y = oy + r * np.sin(alpha)
            alpha += d_alpha
            if within_bounds(x, y, bounds):
                yield (float(x), float(y))
        else:
            alpha = 0.0
            d_alpha += float(angle_step_change)
            r -= st
            if r < 0:
                break
            x = ox + r * np.cos(alpha)
            y = oy + r * np.sin(alpha)
            if within_bounds(x, y, bounds):
                yield (float(x), float(y))


# -------------------------
# Convex hull fill raster
# -------------------------

def iter_convex_hull_fill(
    hull_points: Sequence[TargetXY],
    *,
    xstep: float,
    ystep: float,
    bounds: Optional[Bounds] = None,
    order: str = "xy",
) -> Iterator[TargetXY]:
    """
    Fill a convex hull region with a grid of points and yield those inside the hull.

    This reproduces legacy ConvexHullRaster.preview_path:
    - Create Delaunay triangulation from hull points
    - Create grid spanning bounds (or hull bbox)
    - Yield points where find_simplex >= 0

    order:
      - "xy": iterate x outer, y inner (matches legacy loop order)
      - "yx": iterate y outer, x inner
    """
    if xstep <= 0 or ystep <= 0:
        raise ValueError("xstep and ystep must be > 0")
    if len(hull_points) < 3:
        raise ValueError("Need at least 3 hull points")

    hull_arr = np.array(hull_points, dtype=float)
    tri = Delaunay(hull_arr)

    if bounds is None:
        bounds = bounds_from_points(hull_points)
    xmin, xmax, ymin, ymax = bounds

    xs = arange_inclusive(xmin, xmax, xstep, include_stop=False)
    ys = arange_inclusive(ymin, ymax, ystep, include_stop=False)

    if order not in ("xy", "yx"):
        raise ValueError("order must be 'xy' or 'yx'")

    if order == "xy":
        for x in xs:
            for y in ys:
                pt = np.array([x, y], dtype=float)
                if tri.find_simplex(pt) >= 0:
                    yield (float(x), float(y))
    else:
        for y in ys:
            for x in xs:
                pt = np.array([x, y], dtype=float)
                if tri.find_simplex(pt) >= 0:
                    yield (float(x), float(y))


# -------------------------
# High-level path factory
# -------------------------

@dataclass(frozen=True)
class RasterSpec:
    """
    A serializable description of a raster path (useful for UI + network).
    """
    kind: str  # "square_x" | "square_y" | "spiral" | "hull"
    bounds: Optional[Bounds] = None

    # square params
    xstep: float = 0.01
    ystep: float = 0.01

    # spiral params
    origin: Optional[TargetXY] = None
    radius: float = 1.0
    step: float = 0.1
    angle_step: float = 0.2
    angle_step_change: float = 0.0

    # hull params
    hull_points: Optional[Sequence[TargetXY]] = None
    hull_order: str = "xy"


def iter_path_from_spec(spec: RasterSpec, *, default_origin: Optional[TargetXY] = None) -> Iterator[TargetXY]:
    k = spec.kind.lower().strip()

    if k in ("square_x", "square-raster-x", "squarex"):
        if spec.bounds is None:
            raise ValueError("square_x requires bounds")
        return iter_square_raster_x(spec.bounds, spec.xstep, spec.ystep)

    if k in ("square_y", "square-raster-y", "squarey"):
        if spec.bounds is None:
            raise ValueError("square_y requires bounds")
        return iter_square_raster_y(spec.bounds, spec.xstep, spec.ystep)

    if k in ("spiral", "spiral_raster"):
        origin = spec.origin or default_origin
        if origin is None:
            raise ValueError("spiral requires origin (or default_origin)")
        return iter_spiral_inward(
            origin=origin,
            radius=spec.radius,
            step=spec.step,
            angle_step=spec.angle_step,
            angle_step_change=spec.angle_step_change,
            bounds=spec.bounds,
        )

    if k in ("hull", "convex_hull", "convex-hull"):
        if spec.hull_points is None:
            raise ValueError("hull requires hull_points")
        return iter_convex_hull_fill(
            hull_points=spec.hull_points,
            xstep=spec.xstep,
            ystep=spec.ystep,
            bounds=spec.bounds,
            order=spec.hull_order,
        )

    raise ValueError(f"Unknown raster kind: {spec.kind}")
