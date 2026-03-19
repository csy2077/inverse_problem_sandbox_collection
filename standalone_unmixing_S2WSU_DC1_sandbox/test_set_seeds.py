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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_S2WSU_DC1_sandbox/run_code/std_data/data_set_seeds.pkl'
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
        print("FAIL: No outer data file found for set_seeds.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_result = set_seeds(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: set_seeds(*args, **kwargs) raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_result):
            print(f"FAIL: Expected set_seeds to return a callable, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL (inner data {os.path.basename(inner_path)}): {msg}")
                all_passed = False

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Scenario A: Simple function call
        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            # Additional verification: check that the seed was actually set
            # by verifying numpy random state produces deterministic output
            try:
                seed_val = outer_args[0] if outer_args else outer_kwargs.get('seed', None)
                if seed_val is not None:
                    # Set seed again and generate a random number
                    set_seeds(seed_val)
                    val1 = np.random.random()
                    # Set seed again and generate - should be identical
                    set_seeds(seed_val)
                    val2 = np.random.random()
                    if val1 != val2:
                        print(f"FAIL: Seed not properly set. Got {val1} and {val2} for same seed {seed_val}")
                        sys.exit(1)
            except Exception as e:
                print(f"WARNING: Additional seed verification failed: {e}")
                traceback.print_exc()

            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()