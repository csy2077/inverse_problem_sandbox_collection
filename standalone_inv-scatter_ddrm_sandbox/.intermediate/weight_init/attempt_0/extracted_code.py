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
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddrm_sandbox/run_code/std_data/data_weight_init.pkl']
    
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
        
        print(f"Outer args: {len(outer_args)} positional arguments")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and execute
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print("\n=== Scenario B: Factory/Closure Pattern ===")
            print("Creating operator from outer data...")
            
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: Result from weight_init is not callable. Type: {type(agent_operator)}")
                sys.exit(1)
            
            print(f"Operator created successfully: {type(agent_operator)}")
            
            # Load and execute inner data
            inner_path = inner_paths[0]  # Use first inner path
            print(f"\nLoading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Executing operator with inner args...")
            result = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function
            print("\n=== Scenario A: Simple Function ===")
            print("Executing weight_init directly...")
            
            result = weight_init(*outer_args, **outer_kwargs)
            expected = outer_output
        
    except Exception as e:
        print(f"ERROR: Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Verification
    try:
        print("\n=== Verification ===")
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")
        
        if isinstance(result, torch.Tensor):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        if isinstance(expected, torch.Tensor):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
        
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"\nTEST FAILED")
            print(f"Reason: {msg}")
            sys.exit(1)
        else:
            print(f"\nTEST PASSED")
            print(f"Verification message: {msg}")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()