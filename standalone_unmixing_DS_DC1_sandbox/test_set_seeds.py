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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_DS_DC1_sandbox/run_code/std_data/data_set_seeds.pkl'
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
        print("FAIL: Could not find outer data file (data_set_seeds.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        # set_seeds sets np.random.seed and returns None
        agent_operator = set_seeds(*outer_args, **outer_kwargs)
        print(f"  set_seeds executed successfully, returned: {agent_operator}")
    except Exception as e:
        print(f"FAIL: Error executing set_seeds: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if not callable(agent_operator):
                print(f"FAIL: agent_operator is not callable (type={type(agent_operator)})")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {inner_path}")
    else:
        # Scenario A: Simple function - result from Phase 1 is the result
        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

    # Additional sanity check: verify that set_seeds actually sets numpy random seed
    # by checking reproducibility
    try:
        test_seed = outer_args[0] if outer_args else 42
        set_seeds(test_seed)
        arr1 = np.random.random(10)
        set_seeds(test_seed)
        arr2 = np.random.random(10)
        if not np.allclose(arr1, arr2):
            print("FAIL: set_seeds does not produce reproducible numpy random state")
            sys.exit(1)
        print("Reproducibility check passed.")
    except Exception as e:
        print(f"WARNING: Reproducibility sanity check failed: {e}")
        traceback.print_exc()
        # Don't exit here - primary test already passed

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()