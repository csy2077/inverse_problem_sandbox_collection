import json
from typing import Any, Dict, List, Optional, Tuple, Union


# --- Extracted Dependencies ---

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    def __delattr__(self, name: str) -> None:
        del self[name]

def load_config(json_path):
    """Load configuration from JSON file"""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to EasyDict recursively
    def to_easydict(d):
        if isinstance(d, dict):
            return EasyDict({k: to_easydict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_easydict(v) for v in d]
        else:
            return d
    
    return to_easydict(config_dict)
