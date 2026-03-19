import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_open_url.pkl'
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
        print("ERROR: Could not find outer data file (data_open_url.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        # Phase 1: Create the operator
        try:
            agent_operator = open_url(*outer_args, **outer_kwargs)
            print(f"  Created agent_operator: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR creating agent_operator: {e}")
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
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Execution successful, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR executing agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        # Phase 1: Execute function
        try:
            result = open_url(*outer_args, **outer_kwargs)
            print(f"  Execution successful, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR executing open_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Handle special result types (file-like objects)
        # If the expected output is a string (filename) and result is a string, compare directly
        # If expected is bytes or BytesIO, we need to handle accordingly
        try:
            # If result is a file-like object (BufferedReader or BytesIO), read its content for comparison
            if hasattr(result, 'read') and hasattr(expected, 'read'):
                result_content = result.read()
                expected_content = expected.read()
                passed, msg = recursive_check(expected_content, result_content)
            elif hasattr(result, 'read') and isinstance(expected, bytes):
                result_content = result.read()
                passed, msg = recursive_check(expected, result_content)
            elif isinstance(result, str) and isinstance(expected, str):
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
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()