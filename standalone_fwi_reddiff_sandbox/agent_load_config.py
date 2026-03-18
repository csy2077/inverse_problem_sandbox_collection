import os
import json


# --- Extracted Dependencies ---

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "data_standalone", "standalone_fwi_reddiff.json")
    with open(config_path, 'r') as f:
        return json.load(f)
