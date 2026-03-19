import sys
import os
import dill
import traceback
import numpy as np
import torch

# Import target function
from agent_parse_int_list import parse_int_list

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_fps_sandbox/run_code/std_data/data_parse_int_list.pkl']
    
    # Determine test scenario by analyzing file paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if 'parent_function' in path or 'parent_' in path:
            inner_paths.append(path)
        elif path.endswith('data_parse_int_list.pkl'):
            outer_path = path
    
    if not outer_path:
        print("ERROR: No outer data file found")
        sys.exit(1)
    
    try:
        # Phase 1: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Executing parse_int_list with outer args: {outer_args}, kwargs: {outer_kwargs}")
        agent_operator = parse_int_list(*outer_args, **outer_kwargs)
        
        # Phase 2: Determine scenario and execute
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found")
            
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                
                if not callable(agent_operator):
                    print(f"ERROR: agent_operator is not callable. Type: {type(agent_operator)}")
                    sys.exit(1)
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Executing agent_operator with inner args: {inner_args}, kwargs: {inner_kwargs}")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verification
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}")
                    print(f"Error: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {inner_path}")
        else:
            # Scenario A: Simple Function
            print("Scenario A detected: Simple function test")
            result = agent_operator
            expected = outer_data.get('output')
            
            # Verification
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Error: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
        
        print("\nAll tests completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"TEST FAILED with exception:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()