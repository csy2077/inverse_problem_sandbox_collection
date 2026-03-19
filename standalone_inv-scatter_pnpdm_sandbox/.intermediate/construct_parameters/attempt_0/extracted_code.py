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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pnpdm_sandbox/run_code/std_data/data_construct_parameters.pkl'
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

    # Phase 1: Load outer data and reconstruct
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

    print(f"[INFO] outer_args types: {[type(a).__name__ for a in outer_args]}")
    print(f"[INFO] outer_kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 2: Execute function
    try:
        agent_result = construct_parameters(*outer_args, **outer_kwargs)
        print("[INFO] construct_parameters executed successfully.")
    except Exception as e:
        print(f"FAIL: construct_parameters raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 3: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected with {len(inner_paths)} inner data file(s).")

        if not callable(agent_result):
            print("FAIL: Expected agent_result to be callable for Scenario B, but it is not.")
            sys.exit(1)

        all_passed = True
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
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("[INFO] Inner execution completed successfully.")
            except Exception as e:
                print(f"FAIL: Inner execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"[INFO] Verification passed for {inner_path}.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected (simple function call).")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()