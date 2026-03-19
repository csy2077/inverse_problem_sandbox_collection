import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def save_reconstruction_plot(rec, output_path):
    """Save reconstruction visualization to PNG file."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(rec, cmap='gray')
    ax.set_title('SIRT GPU Reconstruction')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Attenuation')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstruction plot saved: {output_path}")
