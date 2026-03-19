import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_getBufferSize import getBufferSize

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_twodisloc_sandbox/run_code/std_data/data_getBufferSize.pkl'
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
        print("FAIL: No outer data file found for getBufferSize.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct/run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = getBufferSize(*outer_args, **outer_kwargs)
        print("Phase 1: getBufferSize executed successfully.")
    except Exception as e:
        print(f"FAIL: Error executing getBufferSize with outer data args/kwargs.")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from getBufferSize, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully on inner data.")
            except Exception as e:
                print(f"FAIL: Error executing operator with inner data args/kwargs.")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}.")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner data {os.path.basename(inner_path)}: PASSED")
            except Exception as e:
                print(f"FAIL: Error during recursive_check for inner data.")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call.")

        result = agent_result
        expected = outer_data.get('output')

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("  Outer data verification: PASSED")
        except Exception as e:
            print(f"FAIL: Error during recursive_check.")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()