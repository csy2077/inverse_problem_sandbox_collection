import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) categories
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (standard_data / data_recon_slice.pkl) found.")
        sys.exit(1)

    # --- Phase 1: Load outer data and execute recon_slice ---
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_result = recon_slice(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: recon_slice execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"FAIL: Expected recon_slice to return a callable (operator), got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL (inner test {idx + 1}): {msg}")
                all_passed = False
            else:
                print(f"PASS (inner test {idx + 1})")

        if not all_passed:
            sys.exit(1)
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function, comparing output directly.")

        expected = outer_output
        result = agent_result

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
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()