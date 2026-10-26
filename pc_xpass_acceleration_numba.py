"""Compiled point/disk TTA lookup; imported lazily by pc_xpass_acceleration."""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def arrival_corrections(queries, low, high, children, axis, values, fallback):
    """Return normalized corrections and a mask of trusted interpolation rows."""
    result = np.zeros(len(queries))
    valid = np.zeros(len(queries), np.bool_)
    for i in range(len(queries)):
        inside = True
        for dim in range(3):
            coordinate = queries[i, dim]
            if (not math.isfinite(coordinate) or coordinate < low[0, dim]
                    or coordinate > high[0, dim]):
                inside = False
        if not inside:
            continue
        node = 0
        while children[node, 0] >= 0:
            dim = axis[node]
            side = int(queries[i, dim] >= .5*(low[node, dim]+high[node, dim]))
            node = children[node, side]
        # An internal node's fallback flag does not describe its descendants.
        if fallback[node]:
            continue
        x = (queries[i, 0]-low[node, 0])/(high[node, 0]-low[node, 0])
        y = (queries[i, 1]-low[node, 1])/(high[node, 1]-low[node, 1])
        z = (queries[i, 2]-low[node, 2])/(high[node, 2]-low[node, 2])
        total = 0.
        # Same corner order as itertools.product((0, 1), repeat=3).
        for corner in range(8):
            w0 = x if corner & 4 else 1.-x
            w1 = y if corner & 2 else 1.-y
            w2 = z if corner & 1 else 1.-z
            total += ((w0*w1)*w2)*values[node, corner]
        result[i] = total
        valid[i] = True
    return result, valid
