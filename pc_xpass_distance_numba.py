"""Compiled support interpolation and angular maxima for pc-xPass distance.

Imported lazily by pc_xpass_distance. Kernels are single threaded and retain
strict floating-point arithmetic; Numba caches native code between worker runs.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def support_profiles(beta, times, angles, low, high, children, axis, values):
    """Evaluate both one-sided endpoints of every angular interval."""
    width = len(angles) - 1
    left = np.empty((len(times), width))
    right = np.empty_like(left)
    for i in range(len(times)):
        for j in range(width):
            theta = .5 * (angles[j] + angles[j + 1])
            node = 0
            while children[node, 0] >= 0:
                dim = axis[node]
                coordinate = beta if dim == 0 else theta if dim == 1 else times[i]
                side = int(coordinate >= .5 * (low[node, dim] + high[node, dim]))
                node = children[node, side]
            x = (beta - low[node, 0]) / (high[node, 0] - low[node, 0])
            z = (times[i] - low[node, 2]) / (high[node, 2] - low[node, 2])
            v0 = ((1 - x) * ((1 - z) * values[node, 0] + z * values[node, 1])
                  + x * ((1 - z) * values[node, 4] + z * values[node, 5]))
            v1 = ((1 - x) * ((1 - z) * values[node, 2] + z * values[node, 3])
                  + x * ((1 - z) * values[node, 6] + z * values[node, 7]))
            # Use this interval's own leaf on both ends, including hanging
            # adaptive faces whose interpolants may have different limits.
            y0 = (angles[j] - low[node, 1]) / (high[node, 1] - low[node, 1])
            y1 = (angles[j + 1] - low[node, 1]) / (high[node, 1] - low[node, 1])
            left[i, j] = v0 + y0 * (v1 - v0)
            right[i, j] = v0 + y1 * (v1 - v0)
    return left, right


@njit(cache=True)
def angular_maxima(targets, inverse, left, right, angles, cosines, sines):
    """Maximize the same piecewise-linear support interpolant analytically."""
    result = np.empty(len(targets))
    for i in range(len(targets)):
        x, y = targets[i]
        radius = math.hypot(x, y)
        phi = math.atan2(y, x)
        profile = inverse[i]
        best = -np.inf
        for j in range(len(angles) - 1):
            p0 = x * cosines[j] + y * sines[j]
            p1 = x * cosines[j + 1] + y * sines[j + 1]
            best = max(best, p0 - left[profile, j], p1 - right[profile, j])
        for j in range(len(angles) - 1):
            p0 = x * cosines[j] + y * sines[j]
            p1 = x * cosines[j + 1] + y * sines[j + 1]
            upper = radius if angles[j] <= phi <= angles[j + 1] else max(p0, p1)
            if upper - min(left[profile, j], right[profile, j]) <= best:
                continue
            slope = (right[profile, j] - left[profile, j]) / (angles[j + 1] - angles[j])
            if abs(slope) >= radius:
                continue
            theta = phi - math.asin(slope / radius)
            if angles[j] <= theta <= angles[j + 1]:
                value = (x * math.cos(theta) + y * math.sin(theta) - left[profile, j]
                         - slope * (theta - angles[j]))
                best = max(best, value)
        result[i] = best
    return result
