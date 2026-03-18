import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging
from agent_create_logger import create_logger
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddrm_sandbox/run_code/std_data/data_create_logger.pkl']
    
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if basename == 'data_create_logger.pkl':
            outer_path = path
        elif 'parent_function' in basename and 'create_logger' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_create_logger.pkl)")
        sys.exit(1)
    
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Creating operator with args={outer_args}, kwargs={outer_kwargs}")
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        
        if inner_paths:
            print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found")
            
            for inner_path in inner_paths:
                print(f"\nProcessing inner data: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Executing operator with inner args={inner_args}, kwargs={inner_kwargs}")
                result = agent_operator(*inner_args, **inner_kwargs)
                
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}")
                    print(f"Verification message: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
        else:
            print("Scenario A detected: No inner data files, testing direct output")
            result = agent_operator
            expected = outer_data.get('output')
            
            # Logger objects cannot be meaningfully compared after deserialization
            # Verify the result is a logger with expected properties
            if isinstance(result, logging.Logger):
                # Check that logger was created with correct configuration
                if len(result.handlers) == 2:  # Should have console and file handlers
                    handler_types = sorted([type(h).__name__ for h in result.handlers])
                    if 'FileHandler' in handler_types and 'StreamHandler' in handler_types:
                        print("TEST PASSED")
                        sys.exit(0)
                    else:
                        print(f"TEST FAILED: Expected FileHandler and StreamHandler, got {handler_types}")
                        sys.exit(1)
                elif len(result.handlers) == 1 and isinstance(result.handlers[0], logging.NullHandler):
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: Unexpected handler configuration: {len(result.handlers)} handlers")
                    sys.exit(1)
            else:
                print(f"TEST FAILED: Expected logging.Logger, got {type(result)}")
                sys.exit(1)
        
        print("\nAll tests completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR during test execution: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()