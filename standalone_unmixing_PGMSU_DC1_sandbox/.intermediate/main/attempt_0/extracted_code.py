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
    """Test the main function using captured data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_PGMSU_DC1_sandbox/run_code/std_data/data_main.pkl'
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
    
    assert outer_path is not None, f"Could not find outer data file in {data_paths}"
    
    # Phase 1: Load outer data
    print(f"[INFO] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[FAIL] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"[INFO] Outer args count: {len(outer_args)}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")
        
        # Phase 1: Create operator
        print("[INFO] Phase 1: Creating operator via main()")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        print(f"[INFO] Operator created successfully: {type(agent_operator)}")
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[FAIL] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"[INFO] Inner args count: {len(inner_args)}")
            print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[FAIL] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare
            print("[INFO] Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"[FAIL] Comparison raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"[FAIL] Test failed: {msg}")
                sys.exit(1)
            else:
                print("[PASS] Inner test passed.")
    
    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call")
        
        # Phase 1: Execute main
        print("[INFO] Executing main()")
        try:
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] Failed to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Compare
        print("[INFO] Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"[FAIL] Comparison raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"[FAIL] Test failed: {msg}")
            sys.exit(1)
        else:
            print("[PASS] Test passed.")
    
    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    test_main()