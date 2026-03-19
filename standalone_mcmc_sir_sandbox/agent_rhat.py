import numpy as np


# --- Extracted Dependencies ---

def _within(chains):
    """Calculates mean within-chain variance."""
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)

def _between(chains):
    """Calculates mean between-chain variance."""
    n = chains.shape[1]
    within_chain_means = np.mean(chains, axis=1)
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)
    return n * between_chain_var

def rhat(chains, warm_up=0.0):
    """
    Returns the convergence measure R-hat for MCMC chains.
    """
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError('Dimension of chains must be 2 or 3.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError('warm_up only takes values in [0,1].')

    n = chains.shape[1]
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]
    n = n // 2
    if n < 1:
        raise ValueError('Number of samples per chain after warm-up is too small.')
    chains = np.vstack([chains[:, :n], chains[:, -n:]])
    w = _within(chains)
    b = _between(chains)
    return np.sqrt((n - 1.0) / n + b / (w * n))
