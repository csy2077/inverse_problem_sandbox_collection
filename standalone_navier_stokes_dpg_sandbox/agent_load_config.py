import json


# --- Extracted Dependencies ---

def load_config(json_path):
    """Load configuration from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)
