import sys
import os
import dill
import numpy as np
import traceback
from agent_load_data_from_json import load_data_from_json
from verification_utils import recursive_check

def main():
    """
    Robust Unit Test for load_data_from_json
    Handles both Scenario A (simple function) and Scenario B (factory/closure pattern)
    """
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/std_data/data_load_data_from_json.pkl']
    
    print("=" * 80)
    print("UNIT TEST: load_data_from_json")
    print("=" * 80)
    
    # Step 1: Analyze data paths to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is the outer data (exact match for function name)
        if filename == 'data_load_data_from_json.pkl':
            outer_path = path
        # Check if this is inner data (contains parent_function pattern)
        elif 'parent_function' in filename and 'load_data_from_json' in filename:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (data_load_data_from_json.pkl)")
        sys.exit(1)
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    if is_factory_pattern:
        print(f"SCENARIO B: Factory/Closure Pattern Detected")
        print(f"  - Outer Data: {outer_path}")
        print(f"  - Inner Data Files: {len(inner_paths)}")
    else:
        print(f"SCENARIO A: Simple Function")
        print(f"  - Data File: {outer_path}")
    
    print("=" * 80)
    
    try:
        # PHASE 1: Load outer data and reconstruct operator/agent
        print("\n[PHASE 1] Loading outer data and creating operator...")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  Outer args: {len(outer_args)} positional arguments")
        print(f"  Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Execute the function with outer data
        agent_operator = load_data_from_json(*outer_args, **outer_kwargs)
        
        print(f"  Result type: {type(agent_operator)}")
        
        # PHASE 2: Execution & Verification
        if is_factory_pattern:
            print("\n[PHASE 2] Factory Pattern - Executing inner operations...")
            
            # Verify the operator is callable
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            print(f"  Operator is callable: {callable(agent_operator)}")
            
            # Process each inner data file
            all_passed = True
            for idx, inner_path in enumerate(inner_paths, 1):
                print(f"\n  --- Inner Execution {idx}/{len(inner_paths)} ---")
                print(f"  Loading: {os.path.basename(inner_path)}")
                
                try:
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected_output = inner_data.get('output')
                    
                    print(f"    Inner args: {len(inner_args)} positional arguments")
                    print(f"    Inner kwargs: {list(inner_kwargs.keys())}")
                    
                    # Execute the operator with inner data
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
                    
                    # Verify against expected output
                    passed, msg = recursive_check(expected_output, actual_result)
                    
                    if not passed:
                        print(f"    FAILED: {msg}")
                        all_passed = False
                    else:
                        print(f"    PASSED")
                
                except Exception as e:
                    print(f"    ERROR during inner execution: {str(e)}")
                    traceback.print_exc()
                    all_passed = False
            
            if not all_passed:
                print("\n" + "=" * 80)
                print("TEST FAILED: One or more inner executions failed")
                print("=" * 80)
                sys.exit(1)
            
        else:
            # SCENARIO A: Simple function - result is the output
            print("\n[PHASE 2] Simple Function - Verifying output...")
            
            result = agent_operator
            expected = outer_output
            
            print(f"  Result type: {type(result)}")
            print(f"  Expected type: {type(expected)}")
            
            # Verify against expected output
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"\nVERIFICATION FAILED:")
                print(f"  {msg}")
                print("\n" + "=" * 80)
                print("TEST FAILED")
                print("=" * 80)
                sys.exit(1)
        
        # All tests passed
        print("\n" + "=" * 80)
        print("TEST PASSED")
        print("=" * 80)
        sys.exit(0)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()