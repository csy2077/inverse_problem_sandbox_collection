import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import target function
from agent_load_config import load_config

# Import verification utility
from verification_utils import recursive_check


def main():
    # All data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_dpg_sandbox/run_code/std_data/data_load_config.pkl'
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
        print("FAIL: Could not find outer data file (data_load_config.pkl).")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct operator ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_operator = load_config(*outer_args, **outer_kwargs)
        print(f"[INFO] load_config returned: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: Error calling load_config(*args, **kwargs): {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Phase 2: Determine scenario and verify ----
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected. Found {len(inner_paths)} inner data file(s).")

        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected load_config to return a callable, got {type(agent_operator)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"[INFO] Inner data [{idx}] verification passed.")

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        print("[INFO] Scenario A detected. Comparing direct output.")

        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()