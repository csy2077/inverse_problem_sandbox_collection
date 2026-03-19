import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def TilePlot(images, layout, figsize=(10, 10), **kwargs):
    """
    Creates a tiled plot of multiple images.
    
    Args:
        images: tuple of 2D arrays to plot
        layout: (rows, cols) tuple for grid layout
        figsize: figure size tuple
        **kwargs: additional arguments (log_norm, color_scales, etc.)
    
    Returns:
        fig, im, ax: figure, image objects, and axes
    """
    rows, cols = layout
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single axis case
    if rows * cols == 1:
        ax = np.array([ax])
    else:
        ax = ax.flatten()
    
    im = []
    log_norm = kwargs.get('log_norm', False)
    
    for i, img in enumerate(images):
        if i < len(ax):
            # Check if image is complex
            if np.iscomplexobj(img):
                img = np.abs(img)
            
            if log_norm:
                from matplotlib.colors import LogNorm
                # Avoid log(0)
                img_plot = np.where(img > 0, img, img[img > 0].min() if np.any(img > 0) else 1e-10)
                im.append(ax[i].imshow(img_plot, norm=LogNorm()))
            else:
                im.append(ax[i].imshow(img))
    
    return fig, im, ax
