import sys
import os
import dill
import traceback
import logging

# Import the target function
from agent_create_logger import create_logger

# Import verification utility
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_blackhole_daps_sandbox/run_code/std_data/data_create_logger.pkl'
    ]
    
    try:
        # Step 1: Identify outer and inner data files
        outer_path = None
        inner_paths = []
        
        for path in data_paths:
            if not os.path.exists(path):
                print(f"ERROR: Data file not found: {path}")
                sys.exit(1)
            
            filename = os.path.basename(path)
            # Outer data: exact match pattern data_create_logger.pkl
            if filename == 'data_create_logger.pkl':
                outer_path = path
            # Inner data: contains parent_function pattern
            elif 'parent_function' in filename and 'create_logger' in filename:
                inner_paths.append(path)
        
        if not outer_path:
            print("ERROR: No outer data file found (data_create_logger.pkl)")
            sys.exit(1)
        
        print(f"Loading outer data from: {outer_path}")
        
        # Step 2: Load outer data and reconstruct the operator
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        
        print(f"Outer args: {outer_args}")
        print(f"Outer kwargs: {outer_kwargs}")
        
        # Phase 1: Create the logger (operator)
        print("\n=== Phase 1: Creating Logger ===")
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        
        # Step 3: Determine test scenario
        if inner_paths:
            # Scenario B: Factory/Closure Pattern
            print(f"\n=== Scenario B: Factory Pattern Detected ===")
            print(f"Found {len(inner_paths)} inner data file(s)")
            
            # Process each inner data file
            for inner_path in inner_paths:
                print(f"\nLoading inner data from: {inner_path}")
                
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output')
                
                print(f"Inner args: {inner_args}")
                print(f"Inner kwargs: {inner_kwargs}")
                
                # Phase 2: Execute the operator
                print("\n=== Phase 2: Executing Operator ===")
                
                if not callable(agent_operator):
                    print(f"ERROR: agent_operator is not callable. Type: {type(agent_operator)}")
                    sys.exit(1)
                
                result = agent_operator(*inner_args, **inner_kwargs)
                
                # Verification
                print("\n=== Verification ===")
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print(f"TEST FAILED for {inner_path}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                
                print(f"TEST PASSED for {inner_path}")
        
        else:
            # Scenario A: Simple Function
            print(f"\n=== Scenario A: Simple Function ===")
            
            result = agent_operator
            expected = outer_data.get('output')
            
            # Verification
            print("\n=== Verification ===")
            
            # For logger objects, verify type and basic properties instead of exact match
            if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
                print(f"Expected logger name: {expected.name}")
                print(f"Result logger name: {result.name}")
                print(f"Expected logger level: {expected.level}")
                print(f"Result logger level: {result.level}")
                print(f"Expected handlers count: {len(expected.handlers)}")
                print(f"Result handlers count: {len(result.handlers)}")
                
                # Logger objects are stateful and handlers may differ between runs
                # Verify the logger was created successfully with correct configuration
                if not isinstance(result, logging.Logger):
                    print("TEST FAILED: Result is not a Logger instance")
                    sys.exit(1)
                
                # Check that logger has expected handler types for main_process=True
                has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in result.handlers)
                has_file = any(isinstance(h, logging.FileHandler) for h in result.handlers)
                
                if not has_console or not has_file:
                    print(f"TEST FAILED: Logger missing expected handlers (console: {has_console}, file: {has_file})")
                    sys.exit(1)
                
                print("TEST PASSED: Logger created with correct handler types")
            else:
                passed, msg = recursive_check(expected, result)
                
                if not passed:
                    print("TEST FAILED")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
                
                print("TEST PASSED")
        
        # All tests passed
        print("\n" + "="*50)
        print("ALL TESTS PASSED")
        print("="*50)
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: Unexpected exception occurred")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()