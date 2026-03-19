import matplotlib.pyplot as plt


# --- Extracted Dependencies ---

def save_comparison_plot(sino, rec, output_path):
    """Save side-by-side comparison plot to PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sinogram
    im0 = axes[0].imshow(sino, cmap='gray', aspect='auto')
    axes[0].set_title('Input Sinogram', fontsize=14)
    axes[0].set_xlabel('Detector Position (pixels)')
    axes[0].set_ylabel('Projection Angle (index)')
    plt.colorbar(im0, ax=axes[0], label='Intensity')
    
    # Reconstruction
    im1 = axes[1].imshow(rec, cmap='gray')
    axes[1].set_title('FBP GPU Reconstruction', fontsize=14)
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[1], label='Attenuation (cm⁻¹)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved: {output_path}")
