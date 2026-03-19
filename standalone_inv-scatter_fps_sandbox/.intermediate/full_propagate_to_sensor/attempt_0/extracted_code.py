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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_fps_sandbox/run_code/std_data/data_full_propagate_to_sensor.pkl'
    ]
    
    print("=" * 80)
    print("UNIT TEST: full_propagate_to_sensor")
    print("=" * 80)
    
    # Step 1: Analyze data paths to determine test scenario
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
        print(f"SCENARIO B: Factory/Closure Pattern Detected")
        print(f"  - Outer data: {os.path.basename(outer_path)}")
        print(f"  - Inner data files: {len(inner_paths)}")
    else:
        print(f"SCENARIO A: Simple Function")
        print(f"  - Data file: {os.path.basename(outer_path)}")
    
    print("-" * 80)
    
    try:
        # Step 2: Load outer data
        print("Loading outer data...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  - Args count: {len(outer_args)}")
        print(f"  - Kwargs keys: {list(outer_kwargs.keys())}")
        
        # Step 3: Phase 1 - Reconstruct operator/agent
        print("\nPhase 1: Executing full_propagate_to_sensor with outer data...")
        
        try:
            agent_operator = full_propagate_to_sensor(*outer_args, **outer_kwargs)
            print("  ✓ Execution successful")
        except Exception as e:
            print(f"  ✗ Execution failed:")
            print(f"    {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        
        # Step 4: Phase 2 - Execution & Verification
        if is_factory_pattern:
            print("\nPhase 2: Factory Pattern - Testing created operator...")
            
            # Check if agent_operator is callable
            if not callable(agent_operator):
                print(f"  ✗ ERROR: Result is not callable (type: {type(agent_operator).__name__})")
                print("  Expected a callable operator for factory pattern")
                sys.exit(1)
            
            print(f"  ✓ Agent operator is callable")
            
            # Process each inner data file
            all_passed = True
            for idx, inner_path in enumerate(inner_paths, 1):
                print(f"\n  Testing with inner data {idx}/{len(inner_paths)}: {os.path.basename(inner_path)}")
                
                try:
                    # Load inner data
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output')
                    
                    print(f"    - Inner args count: {len(inner_args)}")
                    print(f"    - Inner kwargs keys: {list(inner_kwargs.keys())}")
                    
                    # Execute the agent operator
                    try:
                        result = agent_operator(*inner_args, **inner_kwargs)
                        print(f"    ✓ Operator execution successful")
                    except Exception as e:
                        print(f"    ✗ Operator execution failed:")
                        print(f"      {type(e).__name__}: {str(e)}")
                        traceback.print_exc()
                        all_passed = False
                        continue
                    
                    # Verify result
                    print(f"    Verifying result...")
                    passed, msg = recursive_check(expected, result)
                    
                    if passed:
                        print(f"    ✓ Verification PASSED")
                    else:
                        print(f"    ✗ Verification FAILED:")
                        print(f"      {msg}")
                        all_passed = False
                
                except Exception as e:
                    print(f"    ✗ Error processing inner data:")
                    print(f"      {type(e).__name__}: {str(e)}")
                    traceback.print_exc()
                    all_passed = False
            
            if not all_passed:
                print("\n" + "=" * 80)
                print("TEST FAILED: One or more inner data tests failed")
                print("=" * 80)
                sys.exit(1)
            
            result = agent_operator  # For final summary
            expected = outer_output
            
        else:
            # Scenario A: Simple function - result is already computed
            print("\nPhase 2: Simple Function - Verifying result...")
            result = agent_operator
            expected = outer_output
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"  ✗ Verification FAILED:")
                print(f"    {msg}")
                print("\n" + "=" * 80)
                print("TEST FAILED")
                print("=" * 80)
                sys.exit(1)
            
            print(f"  ✓ Verification PASSED")
        
        # Final success message
        print("\n" + "=" * 80)
        print("TEST PASSED")
        print("=" * 80)
        sys.exit(0)
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR:")
        print(f"  {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()