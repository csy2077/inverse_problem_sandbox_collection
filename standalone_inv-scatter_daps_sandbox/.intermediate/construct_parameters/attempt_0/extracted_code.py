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
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_daps_sandbox/run_code/std_data/data_construct_parameters.pkl'
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
        print("ERROR: No outer data file found for construct_parameters.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Successfully loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 2: Execute the function
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = construct_parameters(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from construct_parameters.")
        except Exception as e:
            print(f"ERROR: Failed to execute construct_parameters (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify agent_operator is callable
        if not callable(agent_operator):
            print("ERROR: agent_operator is not callable. This may not be a factory pattern.")
            sys.exit(1)

        # Load inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Successfully loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner args.")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for inner path: {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for inner path: {os.path.basename(inner_path)}")
                    print(f"Failure message: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        try:
            result = construct_parameters(*outer_args, **outer_kwargs)
            print("Successfully executed construct_parameters.")
        except Exception as e:
            print(f"ERROR: Failed to execute construct_parameters: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare results
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
            else:
                print("TEST FAILED")
                print(f"Failure message: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()