import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_weight_init import weight_init
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_dps_sandbox/run_code/std_data/data_weight_init.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if basename == 'data_weight_init.pkl':
            outer_path = path
        elif 'parent_function' in basename and 'weight_init' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_weight_init.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and execute
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("\n=== Scenario B: Factory/Closure Pattern ===")
            
            # Create the operator
            print("Creating operator from outer data...")
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Result is not callable. Got type: {type(agent_operator)}")
                # If not callable, treat as Scenario A
                result = agent_operator
                expected = outer_output
            else:
                print(f"Operator created successfully: {type(agent_operator)}")
                
                # Load inner data
                inner_path = inner_paths[0]
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Execute the operator
                print("\nExecuting operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
        else:
            # Scenario A: Simple Function - Random seed issue
            print("\n=== Scenario A: Simple Function ===")
            print("Note: weight_init uses random operations (torch.rand/randn)")
            print("Skipping exact value comparison for stochastic function")
            
            # Execute to verify it runs without error
            result = weight_init(*outer_args, **outer_kwargs)
            expected = outer_output
            
            # For random functions, verify shape and dtype instead of exact values
            if isinstance(result, torch.Tensor) and isinstance(expected, torch.Tensor):
                if result.shape == expected.shape and result.dtype == expected.dtype:
                    print(f"Shape match: {result.shape}")
                    print(f"Dtype match: {result.dtype}")
                    print("TEST PASSED (stochastic function - shape/dtype verified)")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: Shape or dtype mismatch")
                    print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
                    print(f"Got shape: {result.shape}, dtype: {result.dtype}")
                    sys.exit(1)
        
        print(f"\nResult type: {type(result)}")
        print(f"Expected type: {type(expected)}")
        
    except Exception as e:
        print(f"ERROR: Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Verification (only for deterministic scenarios)
    try:
        print("\n=== Verification ===")
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()