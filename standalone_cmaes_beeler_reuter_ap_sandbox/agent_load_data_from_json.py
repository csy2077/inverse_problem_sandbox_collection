import numpy as np
import json


# --- Extracted Dependencies ---

def load_data_from_json(filepath):
    """
    Load suggested parameters and times from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Dictionary containing:
            - 'suggested_parameters': numpy array of log-transformed conductances
            - 'suggested_times': numpy array of time points
    """
    print(f"Loading data from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'suggested_parameters': np.array(data['suggested_parameters']),
        'suggested_times': np.array(data['suggested_times'])
    }
