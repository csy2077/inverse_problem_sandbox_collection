```python
import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_open_url import open_url

# Import verification utility
from verification_utils import recursive_check

# Helper function to inject dependencies if needed
def inject_dependencies():
    """Inject any global dependencies required by dill.load"""
    pass

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_blackhole_lgd_sandbox/run_code/std_data/data_open_url.pkl']
    
    # Inject dependencies
    inject_dependencies()
    
    # Determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        # Check if this is outer data (exact match for function name)
        if path.endswith('data_open_url.pkl'):
            outer_path = path
        # Check if this is inner data (contains parent_function pattern)
        elif 'parent_function' in path or 'parent_' in path:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: No outer data file found (data_open_url.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"\nPhase 1: Reconstructing operator with outer data")
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Execute