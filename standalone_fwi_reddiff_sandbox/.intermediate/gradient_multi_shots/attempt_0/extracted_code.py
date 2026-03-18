import sys
import os
import dill
import numpy as np
import traceback

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_gradient_multi_shots import gradient_multi_shots
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_gradient_multi_shots.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for gradient_multi_shots.")
        sys.exit(1)

    # --- Phase 1: Load outer data and run the function ---
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if inner_paths:
        # --- Scenario B: Factory/Closure pattern ---
        print("Scenario B detected: Factory/Closure pattern with inner data files.")

        try:
            print("Running gradient_multi_shots to get operator...")
            agent_operator = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Operator obtained, type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Verify
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for inner data {os.path.basename(inner_path)}")

    else:
        # --- Scenario A: Simple function call ---
        print("Scenario A detected: Simple function call.")

        try:
            print("Running gradient_multi_shots...")
            result = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Result obtained, type: {type(result)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Verify
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()