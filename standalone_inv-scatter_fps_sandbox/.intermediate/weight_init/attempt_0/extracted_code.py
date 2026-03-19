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
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_fps_sandbox/run_code/std_data/data_weight_init.pkl']
    
    # Identify outer and inner paths
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
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[Phase 1] Loading outer data from: {outer_path}")
        print(f"[Phase 1] Outer args: {len(outer_args)} items")
        print(f"[Phase 1] Outer kwargs: {list(outer_kwargs.keys())}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine scenario and execute
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"[Phase 2] Scenario B detected: Factory/Closure pattern")
            print(f"[Phase 2] Found {len(inner_paths)} inner data file(s)")
            
            # Create the operator
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            
            if not callable(agent_operator):
                print(f"ERROR: weight_init did not return a callable. Got: {type(agent_operator)}")
                sys.exit(1)
            
            print(f"[Phase 2] Successfully created operator: {type(agent_operator)}")
            
            # Load and execute with inner data
            inner_path = inner_paths[0]  # Use first inner path
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"[Phase 2] Executing operator with inner data from: {inner_path}")
            print(f"[Phase 2] Inner args: {len(inner_args)} items")
            print(f"[Phase 2] Inner kwargs: {list(inner_kwargs.keys())}")
            
            result = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function
            print(f"[Phase 2] Scenario A detected: Simple function")
            
            result = weight_init(*outer_args, **outer_kwargs)
            expected = outer_data.get('output')
            
            print(f"[Phase 2] Executed weight_init directly")
        
    except Exception as e:
        print(f"ERROR: Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 3: Verification
    try:
        print(f"[Phase 3] Verifying results...")
        print(f"[Phase 3] Result type: {type(result)}")
        print(f"[Phase 3] Expected type: {type(expected)}")
        
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