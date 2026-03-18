import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main_test():
    """Test the main function using captured data."""
    
    # Data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddnm_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Separate outer (main) and inner (parent_function) paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    # ============================================================
    # Phase 1: Load outer data and run main()
    # ============================================================
    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)
    
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # ============================================================
    # Phase 2: Determine scenario and execute
    # ============================================================
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        # Run main to get the operator
        print("Running main() to get operator...")
        try:
            from agent_main import main
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print(f"ERROR: main() did not return a callable. Got: {type(agent_operator)}")
            sys.exit(1)
        
        print(f"Got operator of type: {type(agent_operator)}")
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            print("Executing operator with inner args...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        
        print("Running main()...")
        try:
            from agent_main import main
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == '__main__':
    main_test()