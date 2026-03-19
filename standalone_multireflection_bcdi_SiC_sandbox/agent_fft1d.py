from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t


# --- Extracted Dependencies ---

def fft1d(arr, n):
    """1D FFT along axis n with shifts."""
    return fftshift_t(
        fftn_t(
            fftshift_t(arr, dim=[n]),
            dim=[n],
            norm='ortho'
        ),
        dim=[n]
    )
