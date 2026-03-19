import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def save_sinogram_plot(sinogram, output_path):
    """Save sinogram visualization to PNG file."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sinogram, cmap='gray', aspect='auto')
    ax.set_title('Input Sinogram')
    ax.set_xlabel('Detector Pixel')
    ax.set_ylabel('Projection Angle Index')
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Sinogram plot saved: {output_path}")
