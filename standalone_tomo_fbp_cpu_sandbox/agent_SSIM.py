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

def SSIM(img1, img2, circ_crop=True, L=None, K1=0.01, K2=0.03, sigma=1.5, local_ssim=False):
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except ImportError:
        from skimage.measure import compare_ssim

    if img1.shape != img2.shape:
        raise ValueError('Input images must have same shape.')

    vimg1 = np.zeros(img1.shape)
    vimg2 = np.zeros(img2.shape)

    if circ_crop:
        nrow, ncol = img1.shape
        mask = get_circular_mask(nrow, ncol)
        vimg1[mask] = img1[mask]
        vimg2[mask] = img2[mask]
    else:
        vimg1 = img1
        vimg2 = img2

    val = compare_ssim(vimg1, vimg2, data_range=L, gaussian_weights=True,
                       sigma=sigma, K1=K1, K2=K2, use_sample_covariance=False, full=local_ssim)
    return val
