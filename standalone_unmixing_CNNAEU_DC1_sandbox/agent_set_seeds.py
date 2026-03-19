import numpy as np
import torch


# --- Extracted Dependencies ---

def set_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
