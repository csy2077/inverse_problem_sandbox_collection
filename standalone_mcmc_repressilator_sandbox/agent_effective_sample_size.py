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

def effective_sample_size(samples):
    try:
        n_samples, n_params = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')
    return [effective_sample_size_single_parameter(samples[:, i]) for i in range(n_params)]
