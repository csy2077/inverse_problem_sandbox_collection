import numpy as np
import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def TilePlot(images, layout, figsize=(10, 10), **kwargs):
    """
    Create a tiled plot of multiple images.
    
    Args:
        images: Tuple of 2D arrays to plot
        layout: (rows, cols) tuple
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        fig, im, ax: Figure, image objects, and axes
    """
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
            im.append(ax[i].imshow(img))
            
    return fig, im, ax
