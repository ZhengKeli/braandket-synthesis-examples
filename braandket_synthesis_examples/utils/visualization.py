from re import match
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(matrix: Union[np.ndarray, Any], axes: Optional[plt.Axes] = None, **kwargs):
    nr, nc = np.shape(matrix)

    complex = np.any(np.imag(matrix) != 0)
    if not complex:
        matrix = np.asarray(matrix, dtype=np.float32)
        values = matrix
        d = np.maximum(np.max(np.abs(matrix)), 1.0)
        vmin, vmax = -d, d
    else:
        matrix = np.asarray(matrix, dtype=np.complex64)

        re = np.real(matrix)
        im = np.imag(matrix)
        d = np.maximum(np.maximum(np.max(np.abs(re)), np.max(np.abs(im))), 1.0)

        r = re / d / 2 + 0.5
        g = re / d / 2 + 0.5
        b = im / d / 2 + 0.5

        values = np.stack((r, g, b), axis=-1)
        vmin = vmax = None

    axes: plt.Axes = axes or plt.gca()
    kwargs = {**{"cmap": "coolwarm"}, **kwargs}
    axes.imshow(values, vmin=vmin, vmax=vmax, **kwargs)

    axes.set_axis_off()
    # axes.xaxis.set_ticks_position('top')
    # axes.xaxis.set_ticks(range(nc))
    # axes.yaxis.set_ticks(range(nr))

    for r in range(nr):
        for c in range(nc):
            v = matrix[r, c]
            text = f"{v :.2f}" if v != 0 else ""
            text = text if not match(r"^[+-]?0\.0+$", text) else ""
            axes.text(c, r, text, ha="center", va="center", color="black")
