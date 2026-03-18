import sys
import os
import dill
import torch
import numpy as np
import traceback

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_open_url.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'data_open_url.pkl' or basename.endswith('_open_url.pkl'):
            outer_path = path
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (data_open_url.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Loaded outer data: args={outer_args}, kwargs keys={list(outer_kwargs.keys())}")
    
    # Execute the function
    try:
        agent_result = open_url(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute open_url with outer args/kwargs")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Check if this is a factory pattern (inner paths exist)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Detected factory pattern with {len(inner_paths)} inner data file(s)")
        
        # Verify the agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator but got {type(agent_result)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Loaded inner data from {inner_path}")
            
            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator with inner args/kwargs")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: Verification failed with exception")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(f"Failure message: {msg}")
                sys.exit(1)
            
            print(f"Inner test passed for {inner_path}")
    else:
        # Scenario A: Simple function - the result from Phase 1 is the final result
        print("Detected simple function pattern (no inner data)")
        
        result = agent_result
        expected = outer_output
        
        # Verify result
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: Verification failed with exception")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(f"Failure message: {msg}")
            sys.exit(1)
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()