import json
import numpy as np


# --- Extracted Dependencies ---

def load_data(json_file):
    """
    Load model data and parameters from a JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing model data.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['model_initial_conditions'] = np.array(data['model_initial_conditions'])
    data['suggested_parameters'] = np.array(data['suggested_parameters'])
    data['suggested_times'] = np.array(data['suggested_times'])
    data['mcmc_initial_guesses'] = [list(x) for x in data['mcmc_initial_guesses']]
    
    return data
