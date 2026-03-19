import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_generate_em_functions import generate_em_functions
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pnpdm_sandbox/run_code/std_data/data_generate_em_functions.pkl'
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
        print("FAIL: No outer data file found for generate_em_functions.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
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
    outer_expected = outer_data.get('output', None)

    try:
        agent_result = generate_em_functions(*outer_args, **outer_kwargs)
        print("[INFO] generate_em_functions executed successfully.")
    except Exception as e:
        print(f"FAIL: generate_em_functions raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: inner data files found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from generate_em_functions, got {type(agent_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("[INFO] Inner operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Inner operator execution raised an exception:")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner verification passed for: {inner_path}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception:")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: no inner data files, comparing direct output.")

        result = agent_result
        expected = outer_expected

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Outer verification passed.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception:")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()