import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_weight_init import weight_init

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for weight_init."""
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_weight_init.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'data_weight_init.pkl' or basename.endswith('_weight_init.pkl'):
            outer_path = path
    
    # Scenario A: Simple function test (no inner paths)
    if not inner_paths:
        if outer_path is None:
            print("ERROR: No valid data file found")
            sys.exit(1)
        
        try:
            # Load outer data
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            args = outer_data.get('args', ())
            kwargs = outer_data.get('kwargs', {})
            expected_output = outer_data.get('output')
            
            print(f"Loaded data from: {outer_path}")
            print(f"Args: {args}")
            print(f"Kwargs: {kwargs}")
            
        except Exception as e:
            print(f"ERROR loading outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Execute the function
            result = weight_init(*args, **kwargs)
            print(f"Function executed successfully")
            print(f"Result type: {type(result)}")
            
        except Exception as e:
            print(f"ERROR executing weight_init: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            # Compare results
            passed, msg = recursive_check(expected_output, result)
            
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario B: Factory/Closure pattern (inner paths exist)
    else:
        if outer_path is None:
            print("ERROR: No outer data file found for factory pattern")
            sys.exit(1)
        
        try:
            # Phase 1: Load outer data and create operator
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            
            print(f"Loaded outer data from: {outer_path}")
            print(f"Outer Args: {outer_args}")
            print(f"Outer Kwargs: {outer_kwargs}")
            
            # Create the operator/closure
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: weight_init did not return a callable, got {type(agent_operator)}")
                sys.exit(1)
            
            print(f"Agent operator created successfully: {type(agent_operator)}")
            
        except Exception as e:
            print(f"ERROR in Phase 1 (operator creation): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        all_passed = True
        
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"\nLoaded inner data from: {inner_path}")
                print(f"Inner Args: {inner_args}")
                print(f"Inner Kwargs: {inner_kwargs}")
                
                # Execute the operator
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully")
                
                # Compare results
                passed, msg = recursive_check(expected_output, result)
                
                if passed:
                    print(f"Inner test PASSED for {inner_path}")
                else:
                    print(f"Inner test FAILED for {inner_path}: {msg}")
                    all_passed = False
                    
            except Exception as e:
                print(f"ERROR processing inner data {inner_path}: {e}")
                traceback.print_exc()
                all_passed = False
        
        if all_passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()