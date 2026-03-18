import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_construct_parameters import construct_parameters
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddnm_sandbox/run_code/std_data/data_construct_parameters.pkl'
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
        print("FAIL: Could not find outer data file (data_construct_parameters.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct
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

    # Scenario A: No inner paths, simple function call
    if len(inner_paths) == 0:
        print("Scenario A: Simple function call (no inner/closure data)")

        try:
            result = construct_parameters(*outer_args, **outer_kwargs)
            print("  Function executed successfully.")
        except Exception as e:
            print(f"FAIL: construct_parameters raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_data.get('output')
        if expected is None:
            print("FAIL: No expected output found in outer data.")
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"FAIL: Output mismatch.\n{msg}")
            sys.exit(1)

    else:
        # Scenario B: Factory/Closure pattern
        print("Scenario B: Factory/Closure pattern detected")

        try:
            agent_operator = construct_parameters(*outer_args, **outer_kwargs)
            print("  Outer function executed successfully.")
        except Exception as e:
            print(f"FAIL: construct_parameters (outer) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data. func_name: {inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("  Inner function executed successfully.")
            except Exception as e:
                print(f"FAIL: agent_operator (inner) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = inner_data.get('output')
            if expected is None:
                print("FAIL: No expected output found in inner data.")
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if passed:
                print(f"  Inner test PASSED for {os.path.basename(inner_path)}")
            else:
                print(f"FAIL: Output mismatch for {os.path.basename(inner_path)}.\n{msg}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()