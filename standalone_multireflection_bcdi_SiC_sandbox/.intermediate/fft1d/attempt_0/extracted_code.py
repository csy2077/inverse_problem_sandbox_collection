import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import target function
from agent_fft1d import fft1d

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_SiC_sandbox/run_code/std_data/data_fft1d.pkl'
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
        print("FAIL: No outer data file found for fft1d.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Run outer function to get operator
        try:
            agent_operator = fft1d(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure.")
        except Exception as e:
            print(f"FAIL: Error executing fft1d(*outer_args, **outer_kwargs)")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify operator is callable
        if not callable(agent_operator):
            print("FAIL: The returned operator is not callable. This may not be a closure pattern.")
            sys.exit(1)

        # Phase 2: Execute with inner data
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner args.")
            except Exception as e:
                print(f"FAIL: Error executing operator(*inner_args, **inner_kwargs)")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Result mismatch for inner data: {inner_path}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Execute function
        try:
            result = fft1d(*outer_args, **outer_kwargs)
            print("Successfully executed fft1d(*args, **kwargs).")
        except Exception as e:
            print(f"FAIL: Error executing fft1d(*args, **kwargs)")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Result mismatch.")
            print(f"  Message: {msg}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()