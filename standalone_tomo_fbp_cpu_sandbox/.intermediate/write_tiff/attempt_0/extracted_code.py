import sys
import os
import dill
import traceback
import numpy as np

# Attempt torch import but don't fail if not available
try:
    import torch
except ImportError:
    torch = None

from agent_write_tiff import write_tiff
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_write_tiff.pkl'
    ]

    # Classify paths into outer and inner
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for write_tiff.")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct operator ----
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

    # ---- Determine scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")
        try:
            agent_operator = write_tiff(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not create operator from write_tiff: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: write_tiff did not return a callable. Got: {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL (inner {os.path.basename(inner_path)}): {msg}")
                    all_passed = False
                else:
                    print(f"PASS (inner {os.path.basename(inner_path)})")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        # write_tiff returns None (it writes a file), so we need to handle that
        # We run the function and compare the result with expected output
        try:
            result = write_tiff(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not execute write_tiff: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()