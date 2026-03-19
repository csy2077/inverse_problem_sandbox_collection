import sys
import os
import dill
import traceback

# Import the target function
from agent_parse_int_list import parse_int_list
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_score_mri_sandbox/run_code/std_data/data_parse_int_list.pkl'
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
        print("FAIL: Could not find outer data file (data_parse_int_list.pkl)")
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

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Create the operator
        try:
            agent_operator = parse_int_list(*outer_args, **outer_kwargs)
            print(f"  Created operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from parse_int_list: {e}")
            traceback.print_exc()
            sys.exit(1)

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
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Results mismatch for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                print(f"  Expected: {expected}")
                print(f"  Actual:   {result}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function call")

        # Execute the function
        try:
            result = parse_int_list(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not execute parse_int_list: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Results mismatch")
            print(f"  Message: {msg}")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {result}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()