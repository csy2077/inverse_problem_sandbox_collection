import sys
import os
import dill
import numpy as np
import traceback
from scipy.special import hankel1
from scipy.integrate import dblquad

# Import the target function
from agent_generate_em_functions import generate_em_functions

# Import verification utility
from verification_utils import recursive_check

def main():
    """
    Robust Unit Test for generate_em_functions.
    Handles both Scenario A (simple function) and Scenario B (factory/closure pattern).
    """
    
    # Data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddrm_sandbox/run_code/std_data/data_generate_em_functions.pkl'
    ]
    
    print("=" * 80)
    print("UNIT TEST: generate_em_functions")
    print("=" * 80)
    
    # Step 1: Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is the outer data (standard function call)
        if filename == 'data_generate_em_functions.pkl':
            outer_path = path
        # Check if this is inner data (parent_function pattern)
        elif 'parent_function_generate_em_functions' in filename or 'parent_generate_em_functions' in filename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: Could not find outer data file (data_generate_em_functions.pkl)")
        sys.exit(1)
    
    print(f"\n[Phase 1] Loading outer data from: {outer_path}")
    
    # Step 2: Load outer data and reconstruct the operator/agent
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  Outer args type: {type(outer_args)}")
        print(f"  Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'None'}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Execute the target function with outer data
    print("\n[Phase 2] Executing generate_em_functions with outer data...")
    
    try:
        agent_operator = generate_em_functions(*outer_args, **outer_kwargs)
        print(f"  Result type: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR: Function execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Determine test scenario and verify results
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\n[Scenario B] Detected {len(inner_paths)} inner data file(s)")
        print("  Testing factory/closure pattern...")
        
        # Check if agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            print(f"\n  Processing inner data: {os.path.basename(inner_path)}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"    Inner args type: {type(inner_args)}")
                print(f"    Inner kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else 'None'}")
                
                # Execute the operator with inner data
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify results
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"    FAILED: {msg}")
                    all_passed = False
                else:
                    print(f"    PASSED")
                
            except Exception as e:
                print(f"    ERROR: Inner execution failed: {e}")
                traceback.print_exc()
                all_passed = False
        
        if not all_passed:
            print("\n" + "=" * 80)
            print("TEST FAILED: One or more inner executions failed")
            print("=" * 80)
            sys.exit(1)
        
    else:
        # Scenario A: Simple Function
        print("\n[Scenario A] Simple function test")
        print("  Comparing direct output...")
        
        actual_result = agent_operator
        expected_output = outer_output
        
        # Verify results
        passed, msg = recursive_check(expected_output, actual_result)
        
        if not passed:
            print(f"\nFAILED: {msg}")
            print("=" * 80)
            print("TEST FAILED")
            print("=" * 80)
            sys.exit(1)
    
    # All tests passed
    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()