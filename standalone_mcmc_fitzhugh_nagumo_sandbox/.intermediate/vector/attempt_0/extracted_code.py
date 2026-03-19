import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_vector import vector

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_fitzhugh_nagumo_sandbox/run_code/std_data/data_vector.pkl'
    ]

    # Separate outer (standard) and inner (parent_function) data paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (data_vector.pkl) found in data_paths.")
        sys.exit(1)

    # --- Phase 1: Load outer data and run the function ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"[INFO] outer_args types: {[type(a).__name__ for a in outer_args]}")
    print(f"[INFO] outer_kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_result = vector(*outer_args, **outer_kwargs)
        print(f"[INFO] vector() executed successfully. Result type: {type(agent_result).__name__}")
    except Exception as e:
        print(f"FAIL: vector() raised an exception during execution.")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify the result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected vector() to return a callable (operator), but got {type(agent_result).__name__}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_output = inner_data.get('output', None)

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator executed successfully for inner data [{idx}]. Result type: {type(actual_result).__name__}")
            except Exception as e:
                print(f"FAIL: Operator raised an exception during execution on inner data [{idx}].")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_output, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception for inner data [{idx}].")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"[INFO] Inner data [{idx}] verification passed.")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function verification.")

        result = agent_result
        expected = outer_output

        print(f"[INFO] Expected type: {type(expected).__name__}")
        print(f"[INFO] Actual type: {type(result).__name__}")

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception.")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()