import sys
import os
import dill
import traceback

# Import the target function
from agent_parse_int_list import parse_int_list
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_parse_int_list.pkl'
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

    # Phase 1: Load outer data and reconstruct
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    try:
        agent_result = parse_int_list(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: Error executing parse_int_list with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from parse_int_list, got {type(agent_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for {os.path.basename(inner_path)}")
    else:
        # Scenario A: Simple function
        expected = outer_output
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: Error during recursive_check: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()