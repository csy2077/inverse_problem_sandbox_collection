import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_vector import vector
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_lotka_volterra_sandbox/run_code/std_data/data_vector.pkl'
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
        print("FAIL: No outer data file (data_vector.pkl) found.")
        sys.exit(1)

    # Phase 1: Load outer data
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
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Run outer call to get the operator
        try:
            agent_operator = vector(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator from vector()")
        except Exception as e:
            print(f"FAIL: Phase 1 - Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Phase 1 - Result is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2 [{idx}]: Successfully executed operator")
            except Exception as e:
                print(f"FAIL: Phase 2 [{idx}] - Execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Phase 2 [{idx}] - Verification failed: {msg}")
                    all_passed = False
                else:
                    print(f"Phase 2 [{idx}]: Verification passed")
            except Exception as e:
                print(f"FAIL: Phase 2 [{idx}] - recursive_check error: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Run the function
        try:
            result = vector(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully executed vector()")
        except Exception as e:
            print(f"FAIL: Phase 1 - Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                print(f"  Expected type: {type(expected)}")
                print(f"  Result type: {type(result)}")
                if isinstance(expected, np.ndarray):
                    print(f"  Expected shape: {expected.shape}, dtype: {expected.dtype}")
                if isinstance(result, np.ndarray):
                    print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()