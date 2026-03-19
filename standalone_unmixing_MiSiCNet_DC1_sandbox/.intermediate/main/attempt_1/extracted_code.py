import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch agent_main module before importing - it's missing 'import logging'
import agent_main
if not hasattr(agent_main, 'logging'):
    agent_main.logging = logging

# Re-setup the logger if it failed
if not hasattr(agent_main, 'logger') or agent_main.logger is None:
    agent_main.logger = logging.getLogger(agent_main.__name__)

from agent_main import main
from verification_utils import recursive_check


def test_main():
    """Test the main function using captured standard data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MiSiCNet_DC1_sandbox/run_code/std_data/data_main.pkl'
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
    
    assert outer_path is not None, f"Could not find outer data file (data_main.pkl) in {data_paths}"
    
    # ---- Phase 1: Load outer data and run main ----
    print(f"[INFO] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[FAIL] Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"[INFO] Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    
    # ---- Determine scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected: {len(inner_paths)} inner data file(s) found.")
        
        print("[INFO] Running main(*args, **kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        assert callable(agent_operator), (
            f"[FAIL] Expected main() to return a callable operator, got {type(agent_operator)}"
        )
        print(f"[INFO] Got callable operator: {type(agent_operator)}")
        
        # Process each inner data file
        for inner_path in sorted(inner_paths):
            print(f"\n[INFO] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[FAIL] Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"[INFO] Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            
            print("[INFO] Running agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[FAIL] agent_operator() raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            print("[INFO] Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"[FAIL] recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"[FAIL] Verification failed for {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"[PASS] Inner test passed for {os.path.basename(inner_path)}")
    
    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call.")
        
        print("[INFO] Running main(*args, **kwargs)...")
        try:
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("[INFO] Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"[FAIL] recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"[FAIL] Verification failed: {msg}")
            sys.exit(1)
        else:
            print("[PASS] Result matches expected output.")
    
    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    test_main()