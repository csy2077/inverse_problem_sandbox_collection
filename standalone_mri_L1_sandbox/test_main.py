import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def test_main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (main) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, "ERROR: Could not find outer data file (data_main.pkl)"

    # ---------------------------------------------------------------
    # Phase 1: Load outer data and run main()
    # ---------------------------------------------------------------
    try:
        print(f"[INFO] Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Outer data loaded successfully. Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"[INFO] outer_args count: {len(outer_args)}, outer_kwargs keys: {list(outer_kwargs.keys())}")

    # ---------------------------------------------------------------
    # Determine scenario
    # ---------------------------------------------------------------
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        try:
            print("[INFO] Running main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"[INFO] main() returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from main(), got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute operator
        for inner_path in inner_paths:
            try:
                print(f"[INFO] Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Inner data loaded. Keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                print("[INFO] Running agent_operator(*inner_args, **inner_kwargs)...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator returned: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: agent_operator() raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call.")

        try:
            print("[INFO] Running main(*outer_args, **outer_kwargs)...")
            actual_result = main(*outer_args, **outer_kwargs)
            print(f"[INFO] main() returned: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Comparison
        try:
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Verification passed.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    test_main()