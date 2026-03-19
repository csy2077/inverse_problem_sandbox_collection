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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_sir_sandbox/run_code/std_data/data_vector.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (data_vector.pkl) found.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct
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

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        try:
            agent_operator = vector(*outer_args, **outer_kwargs)
            print(f"  Created operator of type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from vector(): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        all_passed = True
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
            except Exception as e:
                print(f"FAIL: Execution of operator failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Result mismatch for {inner_path}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        expected = outer_data.get('output')

        try:
            result = vector(*outer_args, **outer_kwargs)
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Execution of vector() failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Result mismatch.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()