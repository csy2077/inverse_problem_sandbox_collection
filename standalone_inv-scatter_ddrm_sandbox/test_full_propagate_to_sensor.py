import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_full_propagate_to_sensor import full_propagate_to_sensor

# Import verification utility
from verification_utils import recursive_check


def main():
    """
    Robust Unit Test for full_propagate_to_sensor.
    Handles both Scenario A (simple function) and Scenario B (factory/closure pattern).
    """
    
    # Data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddrm_sandbox/run_code/std_data/data_full_propagate_to_sensor.pkl'
    ]
    
    print("=" * 80)
    print("UNIT TEST: full_propagate_to_sensor")
    print("=" * 80)
    
    # Step 1: Analyze data paths to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is the outer data (exact match)
        if filename == 'data_full_propagate_to_sensor.pkl':
            outer_path = path
        # Check if this is inner data (contains parent_function)
        elif 'parent_function' in filename and 'full_propagate_to_sensor' in filename:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (data_full_propagate_to_sensor.pkl)")
        sys.exit(1)
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    if is_factory_pattern:
        print(f"DETECTED: Factory/Closure Pattern (Scenario B)")
        print(f"  - Outer data: {os.path.basename(outer_path)}")
        print(f"  - Inner data files: {len(inner_paths)}")
    else:
        print(f"DETECTED: Simple Function (Scenario A)")
        print(f"  - Data file: {os.path.basename(outer_path)}")
    
    print("-" * 80)
    
    # Step 2: Load outer data
    try:
        print("Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"  - Args count: {len(outer_args)}")
        print(f"  - Kwargs keys: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data")
        print(f"  Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Phase 1 - Reconstruct operator/agent
    try:
        print("\nPhase 1: Executing full_propagate_to_sensor with outer data...")
        agent_operator = full_propagate_to_sensor(*outer_args, **outer_kwargs)
        print(f"  - Result type: {type(agent_operator)}")
        
        if is_factory_pattern:
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            print("  - Operator is callable (factory pattern confirmed)")
        
    except Exception as e:
        print(f"ERROR: Failed to execute full_propagate_to_sensor")
        print(f"  Exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Phase 2 - Execution & Verification
    if is_factory_pattern:
        # Scenario B: Execute the operator with inner data
        print("\nPhase 2: Executing operator with inner data...")
        
        all_passed = True
        
        for inner_path in inner_paths:
            print(f"\n  Testing with: {os.path.basename(inner_path)}")
            
            try:
                # Load inner data
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"    - Inner args count: {len(inner_args)}")
                print(f"    - Inner kwargs keys: {list(inner_kwargs.keys())}")
                
                # Execute the operator
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify result
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"    FAILED: {msg}")
                    all_passed = False
                else:
                    print(f"    PASSED")
                
            except Exception as e:
                print(f"    ERROR: Exception during inner execution")
                print(f"      Exception: {e}")
                traceback.print_exc()
                all_passed = False
        
        if not all_passed:
            print("\n" + "=" * 80)
            print("TEST FAILED: One or more inner executions failed")
            print("=" * 80)
            sys.exit(1)
        else:
            print("\n" + "=" * 80)
            print("TEST PASSED: All inner executions succeeded")
            print("=" * 80)
            sys.exit(0)
    
    else:
        # Scenario A: The result from Phase 1 IS the final result
        print("\nPhase 2: Verifying result against outer data output...")
        
        try:
            result = agent_operator
            expected = outer_output
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"\nFAILED: {msg}")
                print("=" * 80)
                print("TEST FAILED")
                print("=" * 80)
                sys.exit(1)
            else:
                print(f"\nPASSED")
                print("=" * 80)
                print("TEST PASSED")
                print("=" * 80)
                sys.exit(0)
                
        except Exception as e:
            print(f"\nERROR: Exception during verification")
            print(f"  Exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()