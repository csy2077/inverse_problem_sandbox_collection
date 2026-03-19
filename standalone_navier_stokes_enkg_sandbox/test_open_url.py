import sys
import os
import dill
import traceback

# Attempt to import torch and numpy, but don't fail if not available
try:
    import torch
except ImportError:
    torch = None

try:
    import numpy
except ImportError:
    numpy = None

from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_enkg_sandbox/run_code/std_data/data_open_url.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found (data_open_url.pkl).")
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

        # Phase 1: Create the operator
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
                print(f"  Execution successful. Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Compare
            try:
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
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = open_url(*outer_args, **outer_kwargs)
            print(f"  Execution successful. Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute open_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Handle special cases where output is a file object or BytesIO
        # If expected is a file-like object, we compare the content
        try:
            # If both are file-like objects, read and compare contents
            if hasattr(expected, 'read') and hasattr(result, 'read'):
                expected_content = expected.read()
                result_content = result.read()
                # Reset positions
                if hasattr(expected, 'seek'):
                    expected.seek(0)
                if hasattr(result, 'seek'):
                    result.seek(0)
                passed, msg = recursive_check(expected_content, result_content)
            elif isinstance(expected, str) and isinstance(result, str):
                # Both are filenames
                passed, msg = recursive_check(expected, result)
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


if __name__ == '__main__':
    main()