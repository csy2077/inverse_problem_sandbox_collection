import json


# --- Extracted Dependencies ---

def load_data(json_file):
    """
    Load model data from JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing model data.
    
    Returns
    -------
    dict
        Dictionary containing:
        - model_parameters: dict with initial_condition, E_k, g_max
        - protocol: dict with t_hold, t_step, v_hold, v_step
        - suggested_parameters: list of true model parameters
        - suggested_duration: duration of protocol in ms
        - noise_std: standard deviation of noise
        - random_seed: seed used for noise generation
        - times: list of time points
        - noisy_values: list of noisy measurement values
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
