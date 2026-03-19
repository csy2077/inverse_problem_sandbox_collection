import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def save_reconstruction_plot(rec, output_path):
    """Save reconstruction visualization to PNG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(rec, cmap='gray')
    ax.set_title('FBP GPU Reconstruction', fontsize=14)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Attenuation (cm⁻¹)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstruction plot saved: {output_path}")
