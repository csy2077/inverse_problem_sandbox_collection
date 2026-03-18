import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_main import main
from verification_utils import recursive_check

def main_test():
    """
    Robust Unit Test for main() function.
    Handles both Scenario A (simple function) and Scenario B (factory/closure pattern).
    """
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_blackhole_lgd_sandbox/run_code/std_data/data_main.pkl']
    
    print("=" * 80)
    print("UNIT TEST: test_main.py")
    print("=" * 80)
    
    # Phase 0: Analyze data files
    print("\n[Phase 0] Analyzing data files...")
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if basename == 'data_main.pkl' or basename == 'standard_data_main.pkl':
            outer_path = path
            print(f"  Found OUTER data: {path}")
        elif 'parent_function' in basename and 'main' in basename:
            inner_path = path
            print(f"  Found INNER data: {path}")
    
    if not outer_path:
        print("[ERROR] No outer data file found (expected 'data_main.pkl' or 'standard_data_main.pkl')")
        sys.exit(1)
    
    scenario = "B (Factory/Closure)" if inner_path else "A (Simple Function)"
    print(f"\n[Detected Scenario] {scenario}")
    
    # Phase 1: Load outer data and create operator
    print("\n[Phase 1] Loading outer data and creating operator...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  Outer args type: {type(outer_args)}")
        print(f"  Outer kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")
        
        # Execute main to get the operator/result
        print("  Executing: agent_operator = main(*outer_args, **outer_kwargs)")
        agent_operator = main(*outer_args, **outer_kwargs)
        
        print(f"  Result type: {type(agent_operator)}")
        print(f"  Is callable: {callable(agent_operator)}")
        
    except Exception as e:
        print(f"[ERROR] Phase 1 failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute and verify
    print("\n[Phase 2] Execution and verification...")
    
    if inner_path:
        # Scenario B: Factory/Closure pattern
        print("  [Scenario B] Loading inner data for operator execution...")
        
        if not callable(agent_operator):
            print(f"[ERROR] Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"  Inner args type: {type(inner_args)}")
            print(f"  Inner kwargs keys: {list(inner_kwargs.keys()) if isinstance(inner_kwargs, dict) else 'N/A'}")
            
            # Execute the operator
            print("  Executing: result = agent_operator(*inner_args, **inner_kwargs)")
            result = agent_operator(*inner_args, **inner_kwargs)
            
            print(f"  Result type: {type(result)}")
            
        except Exception as e:
            print(f"[ERROR] Phase 2 (Scenario B) failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Scenario A: Simple function
        print("  [Scenario A] Using outer output as expected result...")
        result = agent_operator
        expected = outer_output
    
    # Phase 3: Verification
    print("\n[Phase 3] Comparing result with expected output...")
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("\n" + "=" * 80)
            print("✓ TEST PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ TEST FAILED")
            print("=" * 80)
            print(f"\nFailure message:\n{msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"[ERROR] Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_test()