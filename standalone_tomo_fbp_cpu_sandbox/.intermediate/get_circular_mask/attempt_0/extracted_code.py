import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_get_circular_mask import get_circular_mask

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_get_circular_mask.pkl'
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
        print("FAIL: No outer data file found for get_circular_mask.")
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
        agent_result = get_circular_mask(*outer_args, **outer_kwargs)
        print("Phase 1: get_circular_mask executed successfully.")
    except Exception as e:
        print(f"FAIL: get_circular_mask raised an exception during execution.")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from get_circular_mask, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner execution [{idx}] succeeded.")
            except Exception as e:
                print(f"FAIL: Inner execution [{idx}] raised an exception.")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data [{idx}].")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"  Verification [{idx}] passed.")
            except Exception as e:
                print(f"FAIL: Verification raised an exception for inner data [{idx}].")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification raised an exception.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()