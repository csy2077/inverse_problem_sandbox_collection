import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging

# Import the target function
from agent_create_logger import create_logger
from verification_utils import recursive_check


def compare_loggers(expected, actual):
    """Compare two logger objects by their configuration rather than identity."""
    if not isinstance(expected, logging.Logger) or not isinstance(actual, logging.Logger):
        return False, f"Type mismatch: expected {type(expected)}, got {type(actual)}"
    
    # Compare logging levels
    if expected.level != actual.level:
        return False, f"Level mismatch: expected {expected.level}, got {actual.level}"
    
    # Compare handler types
    expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
    actual_handler_types = sorted([type(h).__name__ for h in actual.handlers])
    
    if expected_handler_types != actual_handler_types:
        return False, f"Handler types mismatch: expected {expected_handler_types}, got {actual_handler_types}"
    
    return True, "Loggers match"


def custom_check(expected, actual):
    """Custom comparison that handles logger objects specially."""
    if isinstance(expected, logging.Logger) and isinstance(actual, logging.Logger):
        return compare_loggers(expected, actual)
    return recursive_check(expected, actual)


def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_create_logger.pkl']
    
    # Filter paths to find outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_paths.append(path)
        elif basename == 'data_create_logger.pkl' or basename.endswith('_create_logger.pkl'):
            outer_path = path
    
    # If no explicit outer path found, use the first available path
    if outer_path is None and len(data_paths) > 0:
        outer_path = data_paths[0]
    
    if outer_path is None:
        print("ERROR: No data file found for create_logger")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator/result
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Extract args and kwargs from outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    
    # Execute the target function
    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to execute create_logger with outer data")
        print(traceback.format_exc())
        sys.exit(1)
    
    # Determine scenario based on inner paths
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator but got {type(agent_operator)}")
            sys.exit(1)
        
        # Process each inner path
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}")
                print(traceback.format_exc())
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')
            
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner data")
                print(traceback.format_exc())
                sys.exit(1)
            
            # Verify result
            try:
                passed, msg = custom_check(expected, result)
            except Exception as e:
                print(f"ERROR: Failed during recursive_check")
                print(traceback.format_exc())
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for inner data {inner_path}")
                print(msg)
                sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple Function
        # The result from create_logger is the final result
        result = agent_operator
        expected = outer_data.get('output')
        
        # Verify result using custom check for loggers
        try:
            passed, msg = custom_check(expected, result)
        except Exception as e:
            print(f"ERROR: Failed during recursive_check")
            print(traceback.format_exc())
            sys.exit(1)
        
        if not passed:
            print("TEST FAILED")
            print(msg)
            sys.exit(1)
        
        print("TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()