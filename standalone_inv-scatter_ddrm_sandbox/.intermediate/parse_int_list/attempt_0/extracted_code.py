import sys
import os
import dill
import traceback
import numpy as np
import torch

# Import target function
from agent_parse_int_list import parse_int_list

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddrm_sandbox/run_code/std_data/data_parse_int_list.pkl']
    
    # Determine scenario by analyzing file names
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        # Check for exact match (outer data)
        if basename == 'data_parse_int_list.pkl':
            outer_path = path
        # Check for parent_function pattern (inner data)
        elif 'parent_function' in basename and 'parse_int_list' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_parse_int_list.pkl)")
        sys.exit(1)
    
    print(f"[Phase 1] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"[Phase 1] Executing parse_int_list with outer args/kwargs")
        print(f"  Args: {outer_args}")
        print(f"  Kwargs: {outer_kwargs}")
        
        agent_operator = parse_int_list(*outer_args, **outer_kwargs)
        
        print(f"[Phase 1] Result type: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR in Phase 1 (Operator Creation):")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\n[Scenario B] Detected {len(inner_paths)} inner data file(s)")
        
        for inner_path in inner_paths:
            print(f"\n[Phase 2] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"[Phase 2] Executing agent_operator with inner args/kwargs")
                print(f"  Args: {inner_args}")
                print(f"  Kwargs: {inner_kwargs}")
                
                if not callable(agent_operator):
                    print(f"ERROR: agent_operator is not callable (type: {type(agent_operator)})")
                    sys.exit(1)
                
                result = agent_operator(*inner_args, **inner_kwargs)
                
                print(f"[Phase 2] Result: {result}")
                print(f"[Phase 2] Expected: {expected}")
                
            except Exception as e:
                print(f"ERROR in Phase 2 (Execution):")
                print(traceback.format_exc())
                sys.exit(1)
    else:
        # Scenario A: Simple Function
        print(f"\n[Scenario A] No inner data files detected")
        result = agent_operator
        expected = outer_data.get('output')
        
        print(f"[Verification] Result: {result}")
        print(f"[Verification] Expected: {expected}")
    
    # Verification
    print(f"\n[Verification] Comparing results...")
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED:")
            print(msg)
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR during verification:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()