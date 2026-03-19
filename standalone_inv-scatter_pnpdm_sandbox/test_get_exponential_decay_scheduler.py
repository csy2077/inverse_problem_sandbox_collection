import sys
import os
import dill
import traceback

# Import the target function
from agent_get_exponential_decay_scheduler import get_exponential_decay_scheduler
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pnpdm_sandbox/run_code/std_data/data_get_exponential_decay_scheduler.pkl'
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
        print("FAIL: No outer data file found for get_exponential_decay_scheduler.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_operator = get_exponential_decay_scheduler(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: Error calling get_exponential_decay_scheduler with outer args/kwargs.")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario and verify
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable from get_exponential_decay_scheduler, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args/kwargs from {inner_path}.")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check for inner data {inner_path}.")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}.")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function - result from Phase 1 IS the result
        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check for outer data.")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed for get_exponential_decay_scheduler.")
            print(f"Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()