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
        - model_parameters: dict with initial_conditions
        - suggested_parameters: list of model parameters [a, b, c, d]
        - suggested_times: list of time points
        - suggested_values: list of [hare, lynx] population data
        - mcmc_config: MCMC configuration settings
        - prior_config: prior distribution settings
        - hmc_config: HMC-specific settings
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
