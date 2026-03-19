import sys
import os
import dill
import numpy as np
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def load_pkl(path):
    """Load a pickle file using dill."""
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def test_main():
    """Test the main function against recorded standard data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_S2WSU_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Separate outer (main) and inner (parent_function) paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    assert outer_path is not None, f"Could not find outer data file (data_main.pkl) in {data_paths}"
    
    # Phase 1: Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        outer_data = load_pkl(outer_path)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data loaded successfully. func_name={outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        # Run main to get the operator
        try:
            print("Running main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Operator obtained: {type(agent_operator)}")
            assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        except Exception as e:
            print(f"FAILED to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                inner_data = load_pkl(inner_path)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"FAILED to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution complete. Result type: {type(result)}")
            except Exception as e:
                print(f"FAILED to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        
        try:
            print("Running main(*args, **kwargs)...")
            result = main(*outer_args, **outer_kwargs)
            print(f"Execution complete. Result type: {type(result)}")
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        expected = outer_output
        
        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAILED during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    test_main()