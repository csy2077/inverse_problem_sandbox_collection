import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def TilePlot(images, layout, figsize=(10, 10), log_norm=False, color_scales=False, **kwargs):
    """Create tiled plot of multiple images."""
    rows, cols = layout
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    # Handle single axis case
    if rows * cols == 1:
        ax = np.array([ax])
    else:
        ax = ax.flatten()

    im = []
    for i, img in enumerate(images):
        if i < len(ax):
            # Check if image is complex
            if np.iscomplexobj(img):
                img = np.abs(img)
            
            if log_norm:
                img_display = np.log10(np.abs(img) + 1e-10)
            else:
                img_display = img
            
            im_obj = ax[i].imshow(img_display)
            if color_scales:
                plt.colorbar(im_obj, ax=ax[i])
            im.append(im_obj)

    return fig, im, ax
