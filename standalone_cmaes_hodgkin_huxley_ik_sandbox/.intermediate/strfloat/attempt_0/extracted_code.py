import sys
import os
import dill
import traceback
import torch
import numpy as np

# Import the target function
from agent_strfloat import strfloat

# Import verification utility
from verification_utils import recursive_check

def main():
    """
    Robust unit test for strfloat function.
    Handles both simple function and factory/closure patterns.
    """
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_cmaes_hodgkin_huxley_ik_sandbox/run_code/std_data/data_strfloat.pkl']
    
    try:
        # Step 1: Identify outer and inner data files
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            if not os.path.exists(path):
                print(f"ERROR: Data file not found: {path}")
                sys.exit(1)
            
            filename = os.path.basename(path)
            
            # Check if this is the outer data (exact match for function name)
            if filename == 'data_strfloat.pkl':
                outer_path = path
            # Check if this is inner data (contains parent_function pattern)
            elif 'parent_function' in filename and 'strfloat' in filename:
                inner_paths.append(path)
        
        if not outer_path:
            print("ERROR: Could not find outer data file (data_strfloat.pkl)")
            sys.exit(1)
        
        # Step 2: Load outer data and reconstruct operator
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Executing strfloat with outer args: {outer_args}, kwargs: {outer_kwargs}")
        agent_operator = strfloat(*outer_args, **outer_kwargs)
        
        # Step 3: Determine test scenario
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"Detected factory/closure pattern. Found {len(inner_paths)} inner data file(s).")
            
            if not callable(agent_operator):
                print(f"ERROR: Expected callable operator from strfloat, got {type(agent_operator)}")
                sys.exit(1)
            
            # Test with first inner data file
            inner_path = inner_paths[0]
            print(f"Loading inner data from: {inner_path}")
            
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            print(f"Executing operator with inner args: {inner_args}, kwargs: {inner_kwargs}")
            result = agent_operator(*inner_args, **inner_kwargs)
            
        else:
            # Scenario A: Simple Function
            print("Detected simple function pattern.")
            result = agent_operator
            expected = outer_data.get('output')
        
        # Step 4: Verification
        print("Verifying results...")
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    
    except Exception as e:
        print(f"TEST FAILED with exception: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()