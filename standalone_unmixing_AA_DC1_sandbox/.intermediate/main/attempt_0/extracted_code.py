import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def test_main():
    """Test the main function using captured data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_AA_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    assert outer_path is not None, f"Could not find outer data file in {data_paths}"
    
    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Import main
    try:
        from agent_main import main
    except Exception as e:
        print(f"FAILED to import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Execute main
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        # Run main to get the operator
        try:
            print("Running main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        print(f"Got callable operator: {type(agent_operator)}")
        
        # Process each inner path
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
            # Execute operator with inner args
            try:
                print("Running agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Compare
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        
        try:
            print("Running main(*args, **kwargs)...")
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Compare result with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAILED during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == "__main__":
    test_main()