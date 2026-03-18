import sys
import os
import dill
import torch
import numpy as np
import traceback

# Determine data paths and classify them
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_daps_sandbox/run_code/std_data/data_main.pkl']

outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    else:
        outer_path = p

def test_main():
    try:
        from agent_main import main
    except Exception as e:
        print(f"FAIL: Could not import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        from verification_utils import recursive_check
    except Exception as e:
        print(f"FAIL: Could not import recursive_check from verification_utils: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 1: Load outer data
    if outer_path is None:
        print("FAIL: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Scenario determination
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Create operator
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"main() returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: main(*args, **kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: main() did not return a callable. Got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in sorted(inner_paths):
            print(f"\nProcessing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data. Keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"agent_operator() returned: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {inner_path}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for inner data: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            actual_result = main(*outer_args, **outer_kwargs)
            print(f"main() returned: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: main(*args, **kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)
        else:
            print("PASSED for outer data.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    test_main()