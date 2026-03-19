import numpy as np


# --- Extracted Dependencies ---

def get_circular_mask(nrow, ncol, radius=None, center=None):
    """Get a boolean circular mask."""
    if radius is None:
        radius = min(ncol, nrow) / 2
    if center is None:
        yc = ncol / 2.0
        xc = nrow / 2.0
    else:
        yc, xc = center

    ny = np.arange(ncol)
    nx = np.arange(nrow)
    x, y = np.meshgrid(nx, ny)
    mask = ((y - yc + 0.5) ** 2 + (x - xc + 0.5) ** 2) < (radius) ** 2
    return mask
