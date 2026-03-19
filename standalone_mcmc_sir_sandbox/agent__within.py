import numpy as np


# --- Extracted Dependencies ---

def _within(chains):
    """Calculates mean within-chain variance."""
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)
