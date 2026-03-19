import numpy as np


# --- Extracted Dependencies ---

def matrix2d(x):
    """Copies x and returns a 2d read-only NumPy array of floats."""
    x = np.array(x, copy=True, dtype=float)
    if x.ndim == 1:
        x = x.reshape((len(x), 1))
    elif x.ndim != 2:
        raise ValueError('Unable to convert to 2d matrix.')
    x.setflags(write=False)
    return x
