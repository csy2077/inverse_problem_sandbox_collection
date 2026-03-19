import json
import numpy as np


# --- Extracted Dependencies ---

def load_data(json_path):
    """
    Load model data from a JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON data file.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'true_parameters': list of true model parameters [a, b, c]
        - 'times': numpy array of time points
        - 'noisy_values': numpy array of noisy observations (shape: n_times x n_outputs)
        - 'sigma': noise standard deviation
        - 'optimization': dict with 'x0', 'boundaries_lower', 'boundaries_upper'
        - 'mcmc': dict with MCMC configuration settings
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['times'] = np.array(data['times'])
    data['noisy_values'] = np.array(data['noisy_values'])
    data['true_parameters'] = list(data['true_parameters'])
    
    return data
