import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_get_sigmas import get_sigmas

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_csgm_mri_sandbox/run_code/std_data/data_get_sigmas.pkl'
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
        print("FAIL: No outer data file found for get_sigmas.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = get_sigmas(*outer_args, **outer_kwargs)
        print("Phase 1: get_sigmas executed successfully.")
    except Exception as e:
        print(f"FAIL: get_sigmas raised an exception during execution.")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B: Found {len(inner_paths)} inner data file(s).")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected get_sigmas to return a callable, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner execution successful for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Executing the returned operator raised an exception.")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception.")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data: {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                all_passed = False

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A: No inner data files found. Comparing direct output.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()