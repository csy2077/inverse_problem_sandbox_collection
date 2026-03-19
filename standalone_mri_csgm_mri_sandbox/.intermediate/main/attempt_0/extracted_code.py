import sys
import os
import dill
import torch
import numpy as np
import traceback

# Determine data paths and classify them
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_csgm_mri_sandbox/run_code/std_data/data_main.pkl']

outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    else:
        outer_path = p

# This is Scenario A: only standard_data_main.pkl exists (no inner/parent_function files)

def main_test():
    try:
        # Load the outer data
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output', None)
        
        print(f"Function name: {outer_data.get('func_name', 'unknown')}")
        print(f"Number of args: {len(outer_args)}")
        print(f"Kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'none'}")
        
    except Exception as e:
        print(f"FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # Import main from agent_main
        from agent_main import main
        print("Successfully imported main from agent_main")
    except Exception as e:
        print(f"FAILED: Could not import main: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        try:
            print("Running main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Got operator of type: {type(agent_operator)}")
            
            if not callable(agent_operator):
                print(f"FAILED: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
        except Exception as e:
            print(f"FAILED: Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        for inner_path in sorted(inner_paths):
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                
                print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
                print("Executing operator with inner args...")
                
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                from verification_utils import recursive_check
                passed, msg = recursive_check(inner_expected, actual_result)
                
                if not passed:
                    print(f"FAILED: Verification failed for {os.path.basename(inner_path)}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASSED: {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"FAILED: Error processing inner data {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        try:
            print("Running main(*args, **kwargs)...")
            actual_result = main(*outer_args, **outer_kwargs)
            print(f"Got result of type: {type(actual_result)}")
        except Exception as e:
            print(f"FAILED: Could not run main: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            from verification_utils import recursive_check
            passed, msg = recursive_check(expected_output, actual_result)
            
            if not passed:
                print(f"FAILED: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print("PASSED: Verification succeeded")
        except Exception as e:
            print(f"FAILED: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\nTEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main_test()