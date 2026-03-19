import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def save_sinogram_plot(sino, output_path):
    """Save sinogram visualization to PNG."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sino, cmap='gray', aspect='auto')
    ax.set_title('Input Sinogram', fontsize=14)
    ax.set_xlabel('Detector Position (pixels)')
    ax.set_ylabel('Projection Angle (index)')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Sinogram plot saved: {output_path}")
