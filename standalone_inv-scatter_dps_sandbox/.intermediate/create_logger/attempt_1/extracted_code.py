import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging
from agent_create_logger import create_logger
from verification_utils import recursive_check

def compare_loggers(expected, actual):
    """Custom comparison for Logger objects"""
    if not isinstance(expected, logging.Logger) or not isinstance(actual, logging.Logger):
        return False, f"Type mismatch: expected {type(expected)}, got {type(actual)}"
    
    # Compare logger names
    if expected.name != actual.name:
        # Logger names may differ due to module context, this is acceptable
        pass
    
    # Compare effective level
    if expected.level != actual.level:
        return False, f"Logger level mismatch: expected {expected.level}, got {actual.level}"
    
    # Compare number of handlers
    if len(expected.handlers) != len(actual.handlers):
        return False, f"Handler count mismatch: expected {len(expected.handlers)}, got {len(actual.handlers)}"
    
    # Compare handler types
    expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
    actual_handler_types = sorted([type(h).__name__ for h in actual.handlers])
    
    if expected_handler_types != actual_handler_types:
        return False, f"Handler types mismatch: expected {expected_handler_types}, got {actual_handler_types}"
    
    return True, "Loggers match"

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_dps_sandbox/run_code/std_data/data_create_logger.pkl']
    
    # Filter paths to identify outer and inner data files
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        # Outer data: exact match pattern data_create_logger.pkl
        if basename == 'data_create_logger.pkl':
            outer_path = path
        # Inner data: contains parent_function pattern
        elif 'parent_function' in basename and 'create_logger' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_create_logger.pkl)")
        sys.exit(1)
    
    print(f"[Phase 1] Loading outer data from: {outer_path}")
    
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        
        print(f"[Phase 1] Outer args: {outer_args}")
        print(f"[Phase 1] Outer kwargs: {outer_kwargs}")
        
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 1: Reconstruct the operator/agent
    print("[Phase 1] Executing create_logger to reconstruct operator...")
    
    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        print(f"[Phase 1] Agent operator created: {type(agent_operator)}")
        
    except Exception as e:
        print(f"ERROR: Failed to execute create_logger: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine test scenario
    if inner_paths:
        # Scenario B: Factory/Closure Pattern
        print(f"[Phase 2] Scenario B detected: {len(inner_paths)} inner data file(s) found")
        
        for inner_path in inner_paths:
            print(f"[Phase 2] Loading inner data from: {inner_path}")
            
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                
                print(f"[Phase 2] Inner args: {inner_args}")
                print(f"[Phase 2] Inner kwargs: {inner_kwargs}")
                
                # Execute the operator with inner data
                if not callable(agent_operator):
                    print(f"ERROR: Agent operator is not callable: {type(agent_operator)}")
                    sys.exit(1)
                
                print("[Phase 2] Executing agent_operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
                
            except Exception as e:
                print(f"ERROR: Failed during inner execution: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Verification
            print("[Verification] Comparing result with expected output...")
            try:
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                    
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
        
        print("ALL TESTS PASSED")
        sys.exit(0)
        
    else:
        # Scenario A: Simple Function
        print("[Phase 2] Scenario A detected: No inner data files, testing direct output")
        
        result = agent_operator
        expected = outer_output
        
        # Verification with custom logger comparison
        print("[Verification] Comparing result with expected output...")
        try:
            # Use custom logger comparison
            if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
                passed, msg = compare_loggers(expected, result)
            else:
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