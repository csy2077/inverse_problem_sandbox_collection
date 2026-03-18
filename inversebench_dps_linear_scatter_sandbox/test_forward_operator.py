import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_forward_operator import forward_operator
from verification_utils import recursive_check


def main():
    """Main test function for forward_operator."""
    
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_forward_operator.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        basename = os.path.basename(path)
        # Check if this is an inner data file (contains 'parent_function')
        if 'parent_function' in basename:
            inner_paths.append(path)
        else:
            # This is the outer/main data file
            outer_path = path
    
    if outer_path is None:
        print("ERROR: No outer data file found (expected *_forward_operator.pkl)")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data paths: {inner_paths}")
    
    # Phase 1: Load outer data and execute forward_operator
    try:
        print("\n--- Phase 1: Loading outer data ---")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Execute the forward_operator
    try:
        print("\n--- Phase 1: Executing forward_operator ---")
        result = forward_operator(*outer_args, **outer_kwargs)
        print(f"forward_operator executed successfully")
        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        
    except Exception as e:
        print(f"ERROR: forward_operator execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Determine the expected output and actual result based on scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # The result from forward_operator should be callable (an operator)
        print("\n--- Scenario B: Factory/Closure Pattern Detected ---")
        
        if not callable(result):
            print(f"WARNING: Result is not callable. Treating as Scenario A.")
            # Fall back to Scenario A
            expected = outer_output
            actual_result = result
        else:
            agent_operator = result
            print(f"Agent operator obtained: {agent_operator}")
            
            # Process each inner data file
            all_passed = True
            for inner_path in inner_paths:
                try:
                    print(f"\n--- Processing inner data: {inner_path} ---")
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    inner_output = inner_data.get('output', None)
                    
                    print(f"Inner args count: {len(inner_args)}")
                    print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                    # Execute the agent operator with inner data
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    print(f"Agent operator executed successfully")
                    
                    # Compare results
                    passed, msg = recursive_check(inner_output, actual_result)
                    if not passed:
                        print(f"FAILED for {inner_path}: {msg}")
                        all_passed = False
                    else:
                        print(f"PASSED for {inner_path}")
                    
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
    else:
        # Scenario A: Simple Function
        print("\n--- Scenario A: Simple Function ---")
        expected = outer_output
        actual_result = result
    
    # Phase 2: Verification (for Scenario A or fallback)
    try:
        print("\n--- Phase 2: Verification ---")
        print(f"Expected type: {type(expected)}")
        print(f"Actual type: {type(actual_result)}")
        
        if hasattr(expected, 'shape'):
            print(f"Expected shape: {expected.shape}")
        if hasattr(actual_result, 'shape'):
            print(f"Actual shape: {actual_result.shape}")
        
        passed, msg = recursive_check(expected, actual_result)
        
        if passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print(f"\nTEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()