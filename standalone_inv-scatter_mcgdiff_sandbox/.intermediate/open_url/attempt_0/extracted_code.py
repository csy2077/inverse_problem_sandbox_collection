import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_mcgdiff_sandbox/run_code/std_data/data_open_url.pkl'
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
        print("ERROR: No outer data file found for open_url.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct the operator
        try:
            agent_operator = open_url(*outer_args, **outer_kwargs)
            print(f"  Created agent_operator: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Execution succeeded, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = open_url(*outer_args, **outer_kwargs)
            print(f"  Execution succeeded, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute open_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Handle special cases where result is a file-like object
        # The recorded output might be a file-like object (BytesIO or file handle)
        # We need to compare their contents if both are file-like
        try:
            # If both are file-like objects, compare their content
            if hasattr(expected, 'read') and hasattr(result, 'read'):
                expected_pos = expected.tell() if hasattr(expected, 'tell') else 0
                result_pos = result.tell() if hasattr(result, 'tell') else 0
                expected_content = expected.read()
                result_content = result.read()
                # Reset positions
                if hasattr(expected, 'seek'):
                    expected.seek(expected_pos)
                if hasattr(result, 'seek'):
                    result.seek(result_pos)

                if expected_content == result_content:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: File contents differ. Expected {len(expected_content)} bytes, got {len(result_content)} bytes.")
                    sys.exit(1)
            # If both are strings (filenames), compare directly
            elif isinstance(expected, str) and isinstance(result, str):
                if expected == result:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: Expected filename '{expected}', got '{result}'")
                    sys.exit(1)
            else:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()