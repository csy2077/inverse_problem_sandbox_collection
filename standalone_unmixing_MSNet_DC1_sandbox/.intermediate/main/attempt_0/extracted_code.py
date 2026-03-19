import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def test_main():
    """Test the main function using captured data."""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MSNet_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, f"Could not find outer data file among: {data_paths}"

    # ---- Phase 1: Load outer data and run main ----
    print(f"[INFO] Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"[FAIL] Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"[INFO] Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"[INFO] Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # ---- Determine scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        print("[INFO] Running main(*outer_args, **outer_kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"[FAIL] Expected callable operator from main(), got {type(agent_operator)}")
            sys.exit(1)

        print(f"[INFO] Got callable operator: {type(agent_operator)}")

        # Process each inner path
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"[FAIL] Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"[INFO] Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"[INFO] Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            print("[INFO] Running agent_operator(*inner_args, **inner_kwargs)...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"[FAIL] agent_operator() raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            print("[INFO] Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"[FAIL] recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"[FAIL] Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"[PASS] Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("[INFO] Scenario A detected: Simple function call")

        print("[INFO] Running main(*outer_args, **outer_kwargs)...")
        try:
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"[FAIL] main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        print("[INFO] Comparing results...")
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"[FAIL] recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"[FAIL] Verification failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    test_main()