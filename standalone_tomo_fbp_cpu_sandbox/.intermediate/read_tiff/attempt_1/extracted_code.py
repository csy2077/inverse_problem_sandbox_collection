import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_read_tiff import read_tiff
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_read_tiff.pkl'
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
        print("FAIL: Could not find outer data file (data_read_tiff.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        # Run the outer function to get the operator
        try:
            agent_operator = read_tiff(*outer_args, **outer_kwargs)
            print(f"[INFO] Outer function returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error executing outer function read_tiff: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify the operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from read_tiff, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            # Execute the operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator execution returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error executing operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"[INFO] Verification passed for {inner_path}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: Simple function call")

        # Run the function
        try:
            result = read_tiff(*outer_args, **outer_kwargs)
            print(f"[INFO] Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error executing read_tiff: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()