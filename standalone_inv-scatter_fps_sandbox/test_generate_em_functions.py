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
    # Data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_fps_sandbox/run_code/std_data/data_generate_em_functions.pkl'
    ]
    
    print("=" * 80)
    print("Unit Test: generate_em_functions")
    print("=" * 80)
    
    # Analyze data paths to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check for outer data (standard data file)
        if filename == 'data_generate_em_functions.pkl':
            outer_path = path
        # Check for inner data (parent function pattern)
        elif 'parent_function_generate_em_functions' in filename or 'parent_generate_em_functions' in filename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_generate_em_functions.pkl)")
        sys.exit(1)
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    if is_factory_pattern:
        print(f"Detected: Factory/Closure Pattern (Scenario B)")
        print(f"  Outer data: {outer_path}")
        print(f"  Inner data files: {len(inner_paths)}")
    else:
        print(f"Detected: Simple Function (Scenario A)")
        print(f"  Data file: {outer_path}")
    
    print("-" * 80)
    
    try:
        # PHASE 1: Load outer data and create operator/agent
        print("PHASE 1: Loading outer data and creating operator...")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  Outer args type: {type(outer_args)}")
        print(f"  Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'None'}")
        
        # Execute the function with outer data
        print("  Executing generate_em_functions with outer data...")
        agent_operator = generate_em_functions(*outer_args, **outer_kwargs)
        
        print(f"  Result type: {type(agent_operator)}")
        
        # PHASE 2: Execution & Verification
        if is_factory_pattern:
            print("\nPHASE 2: Factory pattern - executing inner operations...")
            
            # For factory pattern, we expect the result to be callable
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                sys.exit(1)
            
            # Process each inner data file
            all_passed = True
            for idx, inner_path in enumerate(inner_paths):
                print(f"\n  Processing inner data {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected_output = inner_data.get('output')
                
                print(f"    Inner args type: {type(inner_args)}")
                print(f"    Inner kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else 'None'}")
                
                # Execute the operator with inner data
                print(f"    Executing operator with inner data...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify result
                print(f"    Verifying result...")
                passed, msg = recursive_check(expected_output, actual_result)
                
                if not passed:
                    print(f"    FAILED: {msg}")
                    all_passed = False
                else:
                    print(f"    PASSED")
            
            if not all_passed:
                print("\n" + "=" * 80)
                print("TEST FAILED: One or more inner executions failed")
                print("=" * 80)
                sys.exit(1)
            
        else:
            # Scenario A: Simple function - result is the final output
            print("\nPHASE 2: Simple function - verifying result...")
            
            result = agent_operator
            expected = outer_output
            
            print(f"  Expected type: {type(expected)}")
            print(f"  Actual type: {type(result)}")
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"\nVERIFICATION FAILED:")
                print(msg)
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
        print(f"\nEXCEPTION OCCURRED:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()