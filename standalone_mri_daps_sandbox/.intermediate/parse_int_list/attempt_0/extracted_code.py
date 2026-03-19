import sys
import os
import dill
import traceback

# Import the target function
from agent_parse_int_list import parse_int_list
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_daps_sandbox/run_code/std_data/data_parse_int_list.pkl']

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
        print("FAIL: No outer data file found for parse_int_list.")
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

    try:
        agent_operator = parse_int_list(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: Error calling parse_int_list with outer args.")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_operator):
            print(f"FAIL: Expected parse_int_list to return a callable (operator), but got {type(agent_operator)}")
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
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args from {inner_path}.")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check for {inner_path}.")
                print(f"Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}.")
                print(f"Message: {msg}")
                print(f"Expected: {expected}")
                print(f"Got: {result}")
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)
    else:
        # Scenario A: Simple function - result from Phase 1 IS the result
        result = agent_operator
        expected = outer_data.get('output')

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check.")
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()