import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def test_main():
    """Test the main function using captured I/O data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_dpg_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Classify paths into outer (main) and inner (parent_function) paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)
    
    # ---- Phase 1: Load outer data ----
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
    
    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else []}")
    
    # ---- Determine Scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")
        
        # Phase 1: Run main to get operator
        print("Running main(*args, **kwargs) to obtain operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: main() execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print(f"ERROR: main() returned non-callable: {type(agent_operator)}")
            sys.exit(1)
        
        print(f"Operator obtained: {type(agent_operator)}")
        
        # Phase 2: Execute with inner data
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
            
            print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else []}")
            
            print("Executing agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
    
    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function call.")
        
        print("Running main(*args, **kwargs)...")
        try:
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: main() execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")


if __name__ == '__main__':
    test_main()
    sys.exit(0)