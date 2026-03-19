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
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_get_circular_mask.pkl'
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
        print("FAIL: Could not find outer data file (data_get_circular_mask.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = get_circular_mask(*outer_args, **outer_kwargs)
        print("Phase 1: get_circular_mask executed successfully.")
    except Exception as e:
        print(f"FAIL: Error executing get_circular_mask: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from get_circular_mask, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: agent_operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function.")

        result = agent_result
        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("PASS: Output verified successfully.")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()