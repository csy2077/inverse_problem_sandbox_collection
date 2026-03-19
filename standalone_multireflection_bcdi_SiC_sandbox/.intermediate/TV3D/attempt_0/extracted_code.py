import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_TV3D import TV3D

# Import verification utility
from verification_utils import recursive_check


def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_SiC_sandbox/run_code/std_data/data_TV3D.pkl'
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
        print("FAIL: No outer data file found (data_TV3D.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Extract outer args and kwargs
    try:
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    except Exception as e:
        print(f"FAIL: Could not extract args/kwargs from outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern.")

        # Run TV3D to get operator
        try:
            agent_operator = TV3D(*outer_args, **outer_kwargs)
            print(f"  TV3D returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: TV3D(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"FAIL: Could not extract args/kwargs from inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Execute operator
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call.")

        # Run TV3D
        try:
            result = TV3D(*outer_args, **outer_kwargs)
            print(f"  TV3D returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: TV3D(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("  Outer test passed.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()