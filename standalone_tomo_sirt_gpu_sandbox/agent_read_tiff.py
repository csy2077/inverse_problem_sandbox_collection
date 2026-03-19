import os
import tifffile


# --- Extracted Dependencies ---

def read_tiff(fname):
    """Read a 2D TIFF image."""
    fname = os.path.abspath(fname)
    img = tifffile.imread(fname)
    return img
