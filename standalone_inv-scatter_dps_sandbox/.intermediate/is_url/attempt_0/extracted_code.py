import sys
import os
import dill
import traceback
import numpy as np
import torch
from agent_is_url import is_url
from verification_utils import recursive_check

def main():
    try:
        # Data paths provided
        data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_dps_sandbox/run_code/std_data/data_is_url.pkl']
        
        # Identify outer and inner paths
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            if 'parent_function' in path or 'parent_' in path:
                inner_paths.append(path)
            elif path.endswith('data_is_url.pkl'):
                outer_path = path
        
        if not outer_path:
            print("ERROR: No outer data file found (data_is_url.pkl)")
            sys.exit(1)
        
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Executing is_url with outer args: {outer_args}, kwargs: {outer_kwargs}")
        agent_operator = is_url(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine scenario and execute
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found")
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator from is_url, got {type(agent_operator)}")
                sys.exit(1)
            
            # Process first inner path
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            
            print(f"Executing agent_operator with inner args: {inner_args}, kwargs: {inner_kwargs}")
            result = agent_operator(*inner_args, **inner_kwargs)
            expected = inner_data['output']
            
        else:
            # Scenario A: Simple Function
            print("Scenario A detected: Simple function execution")
            result = agent_operator
            expected = outer_data['output']
        
        # Phase 3: Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"TEST FAILED WITH EXCEPTION: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()