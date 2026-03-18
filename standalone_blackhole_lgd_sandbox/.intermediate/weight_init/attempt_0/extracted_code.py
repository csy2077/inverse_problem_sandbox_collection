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
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_blackhole_lgd_sandbox/run_code/std_data/data_weight_init.pkl']
    
    # File logic setup: Identify outer and inner paths
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
        
        # Execute weight_init with outer data
        agent_operator = weight_init(*outer_args, **outer_kwargs)
        
        print(f"[Phase 1] Agent operator created: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR in Phase 1 (Operator Creation): {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execution & Verification
    try:
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"[Phase 2] Scenario B detected: {len(inner_paths)} inner data file(s)")
            
            for inner_path in inner_paths:
                print(f"[Phase 2] Processing inner data: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"[Phase 2] Inner args: {len(inner_args)} items")
                print(f"[Phase 2] Inner kwargs: {list(inner_kwargs.keys())}")
                
                # Execute the agent operator
                if not callable(agent_operator):
                    print(f"ERROR: Agent operator is not callable: {type(agent_operator)}")
                    sys.exit(1)
                
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verify result
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}")
                    print(f"Verification message: {msg}")
                    sys.exit(1)
                
                print(f"[Phase 2] Inner data verification passed for {os.path.basename(inner_path)}")
        
        else:
            # Scenario A: Simple Function
            print("[Phase 2] Scenario A detected: Simple function (no inner data)")
            
            result = agent_operator
            expected = outer_data.get('output')
            
            # Verify result
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print("TEST FAILED")
                print(f"Verification message: {msg}")
                sys.exit(1)
        
        print("\n" + "="*50)
        print("TEST PASSED")
        print("="*50)
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR in Phase 2 (Execution & Verification): {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()