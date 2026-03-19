import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_construct_parameters import construct_parameters
from verification_utils import recursive_check

# Helper functions that may be needed for dill.load
from scipy.special import hankel1
from scipy.integrate import dblquad


def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_fps_sandbox/run_code/std_data/data_construct_parameters.pkl']
    
    # Determine test scenario by analyzing data paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if basename == 'data_construct_parameters.pkl':
            outer_path = path
        elif 'parent_function' in basename and 'construct_parameters' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_construct_parameters.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output')
        
        print(f"Executing construct_parameters with args={outer_args}, kwargs={outer_kwargs}")
        agent_operator = construct_parameters(*outer_args, **outer_kwargs)
        
    except Exception as e:
        print(f"ERROR in Phase 1 (Operator Reconstruction):")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Phase 2: Determine execution scenario
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"\nScenario B detected: Factory pattern with {len(inner_paths)} inner execution(s)")
        
        # Check if agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable. Type: {type(agent_operator)}")
            print("This appears to be Scenario A (simple function), but inner paths were found.")
            print("Treating as Scenario A instead.")
            
            # Fall back to Scenario A
            result = agent_operator
            expected = outer_output
            
            print("\nVerifying results...")
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"VERIFICATION FAILED:\n{msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during verification:")
                print(traceback.format_exc())
                sys.exit(1)
        
        # Execute with inner data
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                inner_output = inner_data.get('output')
                
                print(f"Executing agent_operator with inner args={inner_args}, kwargs={inner_kwargs}")
                result = agent_operator(*inner_args, **inner_kwargs)
                expected = inner_output
                
                print("\nVerifying results...")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"VERIFICATION FAILED for {os.path.basename(inner_path)}:")
                    print(msg)
                    sys.exit(1)
                else:
                    print(f"Verification PASSED for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR in Phase 2 (Inner Execution) for {inner_path}:")
                print(traceback.format_exc())
                sys.exit(1)
        
        print("\nALL TESTS PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("\nScenario A detected: Simple function (no inner executions)")
        
        result = agent_operator
        expected = outer_output
        
        print("\nVerifying results...")
        try:
            passed, msg = recursive_check(expected, result)
            
            if not passed:
                print(f"VERIFICATION FAILED:\n{msg}")
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