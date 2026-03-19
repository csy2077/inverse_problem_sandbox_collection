import numpy as np
import sigpy as sp


# --- Extracted Dependencies ---

def get_mvue(kspace, s_maps):
    """Get mvue estimate from coil measurements"""
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))
