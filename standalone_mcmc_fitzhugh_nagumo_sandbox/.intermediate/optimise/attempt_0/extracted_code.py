import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_optimise import optimise
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_fitzhugh_nagumo_sandbox/run_code/std_data/data_optimise.pkl'
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
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Scenario B detected: Factory/Closure pattern")

        # Phase 1: Create the operator
        try:
            agent_operator = optimise(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from optimise()")
        except Exception as e:
            print(f"FAIL: Could not create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify it's callable
        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner data")
            except Exception as e:
                print(f"FAIL: Could not execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(inner_expected, result)
                if passed:
                    print(f"TEST PASSED for inner path: {inner_path}")
                else:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        # Execute the function
        try:
            result = optimise(*outer_args, **outer_kwargs)
            print("Successfully executed optimise()")
        except Exception as e:
            print(f"FAIL: Could not execute optimise: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare result against expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"FAIL: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()