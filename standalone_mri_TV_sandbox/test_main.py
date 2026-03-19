import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def test_main():
    """Test the main function against stored reference data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_TV_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Step 1: Categorize data files
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)
    
    # Step 2: Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data loaded successfully. func_name={outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Execute main function
    try:
        print("Executing main(*args, **kwargs)...")
        actual_result = main(*outer_args, **outer_kwargs)
        print(f"main() executed successfully. Result type: {type(actual_result)}")
    except Exception as e:
        print(f"ERROR: Failed to execute main(): {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")
        
        # Verify the operator is callable
        if not callable(actual_result):
            print(f"ERROR: main() returned a non-callable result of type {type(actual_result)}, but inner data files exist (factory pattern expected).")
            sys.exit(1)
        
        agent_operator = actual_result
        
        # Sort inner paths for deterministic order
        inner_paths.sort()
        
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_expected = inner_data.get('output', None)
                print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
                print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("Executing agent_operator(*inner_args, **inner_kwargs)...")
                inner_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator executed successfully. Result type: {type(inner_result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("Comparing inner result against expected output...")
                passed, msg = recursive_check(inner_expected, inner_result)
                if not passed:
                    print(f"VERIFICATION FAILED for {os.path.basename(inner_path)}: {msg}")
                    all_passed = False
                else:
                    print(f"VERIFICATION PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Verification raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        if not all_passed:
            print("\nTEST FAILED: One or more inner verifications failed.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: No inner data files. Comparing main() output directly.")
        
        try:
            print("Comparing actual result against expected output...")
            passed, msg = recursive_check(outer_output, actual_result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print("\nTEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    test_main()