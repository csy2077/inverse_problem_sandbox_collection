import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_matrix2d import matrix2d

# Import verification utility
from verification_utils import recursive_check

def main():
    """
    Robust unit test for matrix2d following factory/closure pattern detection.
    """
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_beeler_reuter_ap_sandbox/run_code/std_data/data_matrix2d.pkl']
    
    print("=" * 80)
    print("UNIT TEST: matrix2d")
    print("=" * 80)
    
    # Step 1: Analyze data files to determine test strategy
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is outer data (exact match for function name)
        if filename == 'data_matrix2d.pkl':
            outer_path = path
        # Check if this is inner data (contains parent_function pattern)
        elif 'parent_function' in filename and 'matrix2d' in filename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_matrix2d.pkl)")
        sys.exit(1)
    
    print(f"Outer data file: {outer_path}")
    if inner_paths:
        print(f"Inner data files found: {len(inner_paths)}")
        for ip in inner_paths:
            print(f"  - {ip}")
    else:
        print("No inner data files found (Scenario A: Simple Function)")
    
    print("-" * 80)
    
    # Step 2: Load outer data and reconstruct operator
    try:
        print("PHASE 1: Loading outer data and creating operator...")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args type: {type(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'None'}")
        
        # Execute the function with outer data
        agent_operator = matrix2d(*outer_args, **outer_kwargs)
        
        print(f"Operator created successfully")
        print(f"Operator type: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR in Phase 1 (Operator Creation):")
        print(traceback.format_exc())
        sys.exit(1)
    
    print("-" * 80)
    
    # Step 3: Determine test scenario and execute
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("PHASE 2: Executing operator with inner data (Scenario B)...")
            
            # Process each inner data file
            all_passed = True
            for inner_path in inner_paths:
                print(f"\nTesting with: {os.path.basename(inner_path)}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args type: {type(inner_args)}")
                print(f"Inner kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else 'None'}")
                
                # Execute the operator
                if not callable(agent_operator):
                    print(f"ERROR: Operator is not callable for inner execution")
                    all_passed = False
                    continue
                
                result = agent_operator(*inner_args, **inner_kwargs)
                
                print(f"Result type: {type(result)}")
                print(f"Expected type: {type(expected)}")
                
                # Verify result
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for {os.path.basename(inner_path)}:")
                    print(msg)
                    all_passed = False
                else:
                    print(f"✓ Verification passed for {os.path.basename(inner_path)}")
            
            if not all_passed:
                sys.exit(1)
                
        else:
            # Scenario A: Simple Function
            print("PHASE 2: Verifying result (Scenario A)...")
            
            result = agent_operator
            expected = outer_output
            
            print(f"Result type: {type(result)}")
            print(f"Expected type: {type(expected)}")
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print("VERIFICATION FAILED:")
                print(msg)
                sys.exit(1)
            else:
                print("✓ Verification passed")
        
    except Exception as e:
        print(f"ERROR in Phase 2 (Execution/Verification):")
        print(traceback.format_exc())
        sys.exit(1)
    
    print("-" * 80)
    print("TEST PASSED")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()