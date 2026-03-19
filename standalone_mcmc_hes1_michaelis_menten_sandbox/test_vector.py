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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_hes1_michaelis_menten_sandbox/run_code/std_data/data_vector.pkl'
    ]

    # Separate outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (data_vector.pkl).")
        sys.exit(1)

    # --- Phase 1: Load outer data and run vector ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = vector(*outer_args, **outer_kwargs)
        print("Phase 1: vector() executed successfully.")
    except Exception as e:
        print(f"FAIL: vector() raised an exception during Phase 1.")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected vector() to return a callable (operator), got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception.")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception.")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function - result from Phase 1 is the output
        print("Scenario A detected: Simple function call.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception.")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()