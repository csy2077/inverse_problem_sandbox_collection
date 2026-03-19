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

def NRMSE(img, ref, mask='whole'):
    """Compute Normalized Root Mean Square Error."""
    from numpy.linalg import norm

    if img.shape != ref.shape:
        raise ValueError('Input arrays must have same shape.')

    if img.ndim != 2 or ref.ndim != 2:
        raise ValueError('Input arrays must be 2D.')

    if isinstance(mask, str):
        if mask == 'whole' or mask is None:
            vimg = img
            vref = ref
        elif mask == 'circ':
            nrow, ncol = img.shape
            mask_arr = get_circular_mask(nrow, ncol)
            vimg = img[mask_arr]
            vref = ref[mask_arr]
        else:
            raise ValueError('Invalid mask type.')
    elif isinstance(mask, np.ndarray):
        if mask.dtype is not np.dtype('bool'):
            raise ValueError('Mask must be boolean array.')
        vimg = img[mask]
        vref = ref[mask]
    else:
        raise ValueError('Invalid mask type.')

    vimg = vimg.astype(np.float64)
    vref = vref.astype(np.float64)
    val = norm(vimg - vref) / norm(vref)
    return val
