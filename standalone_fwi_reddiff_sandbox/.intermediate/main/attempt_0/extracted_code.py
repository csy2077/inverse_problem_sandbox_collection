import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main_test():
    """Test the main function from agent_main.py"""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Separate outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer data loaded. func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 2: Import and run main
    try:
        from agent_main import main
    except Exception as e:
        print(f"ERROR: Failed to import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")
        print("Running main(*args, **kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: main() returned a non-callable object of type {type(agent_operator)}")
            sys.exit(1)

        print(f"main() returned callable: {type(agent_operator)}")

        # Process each inner path
        for inner_path in sorted(inner_paths):
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"Inner data loaded. func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            print("Executing agent_operator(*inner_args, **inner_kwargs)...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed with exception: {e}")
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
        print("Detected Scenario A (Simple function)")
        print("Running main(*args, **kwargs)...")
        try:
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Comparison
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == '__main__':
    main_test()