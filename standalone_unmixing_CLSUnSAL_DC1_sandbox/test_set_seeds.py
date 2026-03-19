import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_set_seeds import set_seeds
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data/data_set_seeds.pkl'
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
        print("FAIL: Could not find outer data file (data_set_seeds.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_operator = set_seeds(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: Error calling set_seeds with outer args/kwargs.")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"FAIL: set_seeds returned a non-callable result in factory pattern. Got: {type(agent_operator)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args/kwargs from {inner_path}.")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception for {inner_path}.")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}.")
                print(f"Message: {msg}")
                all_passed = False

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        result = agent_operator
        expected = outer_output

        # set_seeds returns None (it just sets the numpy random seed)
        # We need to verify the side effect: that np.random.seed was called correctly
        # First, try standard comparison
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception.")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)

        # Additional verification: ensure the seed was actually set correctly
        # by re-running with the same seed and checking deterministic behavior
        try:
            seed_val = outer_args[0] if outer_args else outer_kwargs.get('seed', None)
            if seed_val is not None:
                set_seeds(seed_val)
                rand_vals_1 = np.random.rand(5)
                set_seeds(seed_val)
                rand_vals_2 = np.random.rand(5)
                if not np.array_equal(rand_vals_1, rand_vals_2):
                    print("FAIL: set_seeds did not properly set numpy random seed (non-deterministic results).")
                    sys.exit(1)
        except Exception as e:
            print(f"WARNING: Could not verify seed determinism: {e}")
            # Don't fail on this extra check if the main check passed

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()