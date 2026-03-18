import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_is_url import is_url

# Import verification utility
from verification_utils import recursive_check


def main():
    """Main test function for is_url."""
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_is_url.pkl']
    
    # Analyze data paths to determine test strategy
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_path = path
        elif filename == 'data_is_url.pkl' or filename == 'standard_data_is_url.pkl':
            outer_path = path
        elif 'is_url' in filename and 'parent_function' not in filename:
            outer_path = path
    
    # Ensure we have at least the outer path
    if outer_path is None:
        print("ERROR: Could not find outer data file for is_url")
        sys.exit(1)
    
    print(f"Outer path: {outer_path}")
    print(f"Inner path: {inner_path}")
    
    # Phase 1: Load outer data and potentially reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Successfully loaded outer data")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")
    
    # Scenario B: Factory/Closure Pattern
    if inner_path is not None:
        # Phase 1: Create the operator/closure
        try:
            agent_operator = is_url(*outer_args, **outer_kwargs)
            print(f"Created agent operator: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Agent operator is not callable: {type(agent_operator)}")
            sys.exit(1)
        
        # Phase 2: Load inner data and execute
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print("Successfully loaded inner data")
        except Exception as e:
            print(f"ERROR: Failed to load inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        expected = inner_data.get('output', None)
        
        print(f"Inner args: {inner_args}")
        print(f"Inner kwargs: {inner_kwargs}")
        
        # Execute the operator with inner args
        try:
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"Execution result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Scenario A: Simple Function
    else:
        # Direct function call
        try:
            result = is_url(*outer_args, **outer_kwargs)
            print(f"Function result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute is_url: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
    
    # Phase 3: Comparison
    print(f"Expected type: {type(expected)}")
    print(f"Result type: {type(result)}")
    print(f"Expected value: {expected}")
    print(f"Result value: {result}")
    
    try:
        passed, msg = recursive_check(expected, result)
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if not passed:
        print(f"TEST FAILED: {msg}")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()