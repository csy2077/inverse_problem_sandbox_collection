import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_effective_sample_size_single_parameter import effective_sample_size_single_parameter

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_sir_sandbox/run_code/std_data/data_effective_sample_size_single_parameter.pkl'
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
        print("FAIL: No outer data file found.")
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

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct the operator
        try:
            agent_operator = effective_sample_size_single_parameter(*outer_args, **outer_kwargs)
            print(f"Created agent_operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute agent_operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data verification succeeded for {inner_path}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Execute the function
        try:
            result = effective_sample_size_single_parameter(*outer_args, **outer_kwargs)
            print(f"Function returned result of type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute effective_sample_size_single_parameter: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            print(f"  Expected type: {type(expected)}, Result type: {type(result)}")
            try:
                print(f"  Expected value: {expected}")
                print(f"  Actual value: {result}")
            except Exception:
                pass
            sys.exit(1)
        else:
            print("PASS: Verification succeeded.")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()