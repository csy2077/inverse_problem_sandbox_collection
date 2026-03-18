import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_generate_em_functions import generate_em_functions
from verification_utils import recursive_check


def main():
    """Main test function for generate_em_functions."""
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_generate_em_functions.pkl']
    
    # Determine file types
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'data_generate_em_functions.pkl':
            outer_path = path
    
    # Scenario A: Simple function (no inner paths)
    if outer_path and not inner_paths:
        print(f"Scenario A: Simple function test")
        print(f"Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected_output = outer_data.get('output')
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Execute the function
        try:
            result = generate_em_functions(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAILED: Comparison failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    
    # Scenario B: Factory/Closure pattern
    elif outer_path and inner_paths:
        print(f"Scenario B: Factory/Closure pattern test")
        print(f"Loading outer data from: {outer_path}")
        
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
        except Exception as e:
            print(f"FAILED: Could not load outer data file: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Extract outer args and kwargs
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args count: {len(outer_args)}")
        print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
        
        # Phase 1: Reconstruct the operator
        try:
            agent_operator = generate_em_functions(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: Operator creation failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"FAILED: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)
        
        print(f"Operator created successfully: {type(agent_operator)}")
        
        # Phase 2: Test each inner data file
        for inner_path in inner_paths:
            print(f"Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Extract inner args and kwargs
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected_output = inner_data.get('output')
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute the operator
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare results
            try:
                passed, msg = recursive_check(expected_output, result)
            except Exception as e:
                print(f"FAILED: Comparison failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"FAILED: {msg}")
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    
    else:
        print(f"FAILED: No valid data files found in paths: {data_paths}")
        sys.exit(1)


if __name__ == "__main__":
    main()