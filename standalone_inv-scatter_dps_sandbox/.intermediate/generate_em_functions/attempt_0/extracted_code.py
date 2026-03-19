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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_dps_sandbox/run_code/std_data/data_generate_em_functions.pkl'
    ]
    
    print("=" * 80)
    print("UNIT TEST: generate_em_functions")
    print("=" * 80)
    
    # Step 1: Analyze data paths to determine test scenario
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        if not os.path.exists(path):
            print(f"ERROR: Data file not found: {path}")
            sys.exit(1)
        
        filename = os.path.basename(path)
        
        # Check if this is the outer data (standard data file)
        if filename == 'data_generate_em_functions.pkl':
            outer_path = path
        # Check if this is inner data (parent_function pattern)
        elif 'parent_function' in filename and 'generate_em_functions' in filename:
            inner_paths.append(path)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file (data_generate_em_functions.pkl)")
        sys.exit(1)
    
    # Determine scenario
    is_factory_pattern = len(inner_paths) > 0
    
    if is_factory_pattern:
        print(f"Detected: SCENARIO B (Factory/Closure Pattern)")
        print(f"  Outer data: {outer_path}")
        print(f"  Inner data files: {len(inner_paths)}")
    else:
        print(f"Detected: SCENARIO A (Simple Function)")
        print(f"  Data file: {outer_path}")
    
    print("-" * 80)
    
    try:
        # Phase 1: Load outer data and reconstruct operator/agent
        print("\n[PHASE 1] Loading outer data and creating operator...")
        
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"  Outer args type: {type(outer_args)}")
        print(f"  Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'None'}")
        
        # Execute the function with outer data
        print(f"  Executing: generate_em_functions(*outer_args, **outer_kwargs)")
        agent_operator = generate_em_functions(*outer_args, **outer_kwargs)
        
        print(f"  Result type: {type(agent_operator)}")
        
        # Phase 2: Execute and verify based on scenario
        if is_factory_pattern:
            print("\n[PHASE 2] Factory pattern detected - executing inner operations...")
            
            # Check if agent_operator is callable
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
                print("This might be Scenario A instead. Treating as simple function.")
                is_factory_pattern = False
                result = agent_operator
                expected = outer_output
            else:
                print(f"  Operator is callable: {callable(agent_operator)}")
                
                # Process each inner data file
                all_passed = True
                for idx, inner_path in enumerate(inner_paths):
                    print(f"\n  Processing inner data {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)}")
                    
                    with open(inner_path, 'rb') as f:
                        inner_data = dill.load(f)
                    
                    inner_args = inner_data.get('args', ())
                    inner_kwargs = inner_data.get('kwargs', {})
                    expected = inner_data.get('output')
                    
                    print(f"    Inner args type: {type(inner_args)}")
                    print(f"    Inner kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else 'None'}")
                    
                    # Execute the operator with inner data
                    print(f"    Executing: agent_operator(*inner_args, **inner_kwargs)")
                    result = agent_operator(*inner_args, **inner_kwargs)
                    
                    print(f"    Result type: {type(result)}")
                    
                    # Verify result
                    print(f"    Verifying result against expected output...")
                    passed, msg = recursive_check(expected, result)
                    
                    if not passed:
                        print(f"\n    VERIFICATION FAILED for inner data {idx + 1}:")
                        print(f"    {msg}")
                        all_passed = False
                    else:
                        print(f"    ✓ Inner data {idx + 1} verification passed")
                
                if not all_passed:
                    print("\n" + "=" * 80)
                    print("TEST FAILED: One or more inner data verifications failed")
                    print("=" * 80)
                    sys.exit(1)
                else:
                    print("\n" + "=" * 80)
                    print("TEST PASSED: All verifications successful")
                    print("=" * 80)
                    sys.exit(0)
        
        if not is_factory_pattern:
            print("\n[PHASE 2] Simple function - verifying result...")
            result = agent_operator
            expected = outer_output
            
            print(f"  Result type: {type(result)}")
            print(f"  Expected type: {type(expected)}")
            
            # Verify result
            print(f"  Verifying result against expected output...")
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"\nVERIFICATION FAILED:")
                print(f"{msg}")
                print("\n" + "=" * 80)
                print("TEST FAILED")
                print("=" * 80)
                sys.exit(1)
            else:
                print(f"  ✓ Verification passed")
                print("\n" + "=" * 80)
                print("TEST PASSED")
                print("=" * 80)
                sys.exit(0)
    
    except Exception as e:
        print(f"\nEXCEPTION OCCURRED:")
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("TEST FAILED: Exception during execution")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    main()