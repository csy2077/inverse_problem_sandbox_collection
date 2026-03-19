import os
import json


# --- Extracted Dependencies ---

def load_data(json_path=None):
    """
    Load model data and configuration from JSON file.
    
    Parameters
    ----------
    json_path : str, optional
        Path to the JSON data file. If None, uses the default path based on script location.
    
    Returns
    -------
    dict
        Dictionary containing all model data and configuration.
    """
    if json_path is None:
        # Default: same directory as the script, same name with .json extension
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'data', 'standalone_mcmc_goodwin_oscillator.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data
