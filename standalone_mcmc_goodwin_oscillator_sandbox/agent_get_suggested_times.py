import numpy as np


# --- Extracted Dependencies ---

def get_suggested_times(data):
    """
    Generate suggested times array from data configuration.
    
    Parameters
    ----------
    data : dict
        Data dictionary loaded from JSON file.
    
    Returns
    -------
    np.ndarray
        Array of time points.
    """
    times_config = data['suggested_times']
    return np.linspace(
        times_config['start'],
        times_config['end'],
        times_config['n_points']
    )
