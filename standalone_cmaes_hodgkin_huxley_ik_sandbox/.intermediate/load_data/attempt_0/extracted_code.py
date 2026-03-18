import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_load_data import load_data

# Import verification utility
from verification_utils import recursive_check

def main():
    """
    Robust unit test for load_data function.
    Handles both simple function and factory/closure patterns.
    """
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_hodgkin_huxley_ik_sandbox/run_code/std_data/data_load_data.pkl']
    
    print("=" * 80)
    print("UNIT TEST: load_data")
    print("=" * 80)
    
    # Step 1: Analyze data files to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is the outer data file (exact match)
        if filename == 'data_load_data.pkl':
            outer_path = path
        # Check if this is inner data (contains parent_function)
        elif 'parent_function' in filename and 'load_data' in filename:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (data_load_data.pkl)")
        sys.exit(1)
    
    print(f"Outer data file: {outer_path}")
    if inner_paths:
        print(f"Inner data files: {inner_paths}")
        print("Detected: Factory/Closure Pattern (Scenario B)")
    else:
        print("Detected: Simple Function (Scenario A)")
    
    print("-" * 80)
    
    # Step 2: Load outer data and reconstruct operator
    try:
        print("Phase 1: Loading outer data and creating operator...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  Outer args: {outer_args}")
        print(f"  Outer kwargs: {outer_kwargs}")
        
        # Execute the function to get the operator/result
        agent_operator = load_data(*outer_args, **outer_kwargs)
        print(f"  Operator created: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR in Phase 1 (Operator Creation):")
        print(traceback.format_exc())
        sys.exit(1)
    
    print("-" * 80)
    
    # Step 3: Determine execution strategy based on scenario
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print("Phase 2: Executing operator with inner data...")
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\n  Processing: {os.path.basename(inner_path)}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"    Inner args: {inner_args}")
                print(f"    Inner kwargs: {inner_kwargs}")
                
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
                print(f"    ERROR during execution:")
                print(traceback.format_exc())
                all_passed = False
        
        if not all_passed:
            print("\n" + "=" * 80)
            print("TEST FAILED")
            print("=" * 80)
            sys.exit(1)
            
    else:
        # Scenario A: Simple Function
        print("Phase 2: Verifying simple function result...")
        
        try:
            result = agent_operator
            expected = outer_output
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"FAILED: {msg}")
                print("\n" + "=" * 80)
                print("TEST FAILED")
                print("=" * 80)
                sys.exit(1)
            else:
                print("PASSED")
                
        except Exception as e:
            print(f"ERROR during verification:")
            print(traceback.format_exc())
            sys.exit(1)
    
    # All tests passed
    print("\n" + "=" * 80)
    print("TEST PASSED")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()