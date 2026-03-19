import os
import scipy.io as sio


# --- Extracted Dependencies ---

log = logging.getLogger(__name__)

def save_estimates(Ehat, Ahat, H, W, output_dir):
    """Save estimates to .mat file."""
    data = {"E": Ehat, "A": Ahat.reshape(-1, H, W)}
    filepath = os.path.join(output_dir, "estimates.mat")
    sio.savemat(filepath, data)
    log.info(f"Estimates saved to {filepath}")
