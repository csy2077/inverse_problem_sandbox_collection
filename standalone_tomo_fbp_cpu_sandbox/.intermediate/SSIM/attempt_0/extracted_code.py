import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_SSIM import SSIM
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_SSIM.pkl']

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
        print("FAIL: No outer data file found (data_SSIM.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
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
            agent_operator = SSIM(*outer_args, **outer_kwargs)
            print(f"Created operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from SSIM: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Created operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Operator execution successful. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        try:
            result = SSIM(*outer_args, **outer_kwargs)
            print(f"Function execution successful. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print("Verification passed")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()