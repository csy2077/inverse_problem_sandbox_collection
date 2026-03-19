from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t


# --- Extracted Dependencies ---

def ifft1d(arr, n):
    """1D inverse FFT along axis n with shifts."""
    return fftshift_t(
        ifftn_t(
            fftshift_t(arr, dim=[n]),
            dim=[n],
            norm='ortho'
        ),
        dim=[n]
    )
