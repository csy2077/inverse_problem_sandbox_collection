import numpy as np


# --- Extracted Dependencies ---

def autocorrelation(x):
    """Calculates autocorrelation for a vector x."""
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]

def _autocorrelate_negative(autocorrelation):
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)

def effective_sample_size_single_parameter(x):
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess
