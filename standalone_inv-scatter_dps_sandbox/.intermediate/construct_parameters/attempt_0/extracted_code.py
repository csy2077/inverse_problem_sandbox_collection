import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_construct_parameters import construct_parameters
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_dps_sandbox/run_code/std_data/data_construct_parameters.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for construct_parameters.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"[INFO] outer_args count: {len(outer_args)}")
    print(f"[INFO] outer_kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 2: Execute the function
    try:
        result = construct_parameters(*outer_args, **outer_kwargs)
        print("[INFO] construct_parameters executed successfully.")
    except Exception as e:
        print(f"FAIL: construct_parameters raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 3: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: inner data files found.")
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            # The result from Phase 1 should be callable
            if not callable(result):
                print("FAIL: Expected construct_parameters to return a callable (operator), but it is not callable.")
                sys.exit(1)

            try:
                actual_result = result(*inner_args, **inner_kwargs)
                print("[INFO] Inner operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Verification passed for inner data {os.path.basename(inner_path)}.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly
        print("[INFO] Scenario A detected: simple function, comparing output directly.")

        if expected_output is None:
            print("FAIL: No expected output found in outer data.")
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Verification passed for direct output comparison.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()