import json
import numpy as np


# --- Extracted Dependencies ---

def load_data(json_path):
    """
    Load model data from JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON data file.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists to numpy arrays where appropriate
    data['suggested_parameters'] = np.array(data['suggested_parameters'])
    data['suggested_times'] = np.array(data['suggested_times'])
    data['suggested_values'] = np.array(data['suggested_values'])
    data['model_config']['fixed_parameters'] = np.array(data['model_config']['fixed_parameters'])
    data['model_config']['initial_conditions'] = np.array(data['model_config']['initial_conditions'])
    
    # Generate smooth times from config
    st_config = data['smooth_times']
    data['smooth_times_array'] = np.linspace(
        st_config['start'], 
        st_config['end'], 
        st_config['num_points']
    )
    
    return data
