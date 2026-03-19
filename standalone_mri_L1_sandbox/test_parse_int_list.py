import sys
import os
import dill
import traceback

# Import the target function
from agent_parse_int_list import parse_int_list
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/std_data/data_parse_int_list.pkl']

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
        print("FAIL: Could not find outer data file (standard_data_parse_int_list.pkl or data_parse_int_list.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"[INFO] Outer args: {outer_args}")
    print(f"[INFO] Outer kwargs: {outer_kwargs}")
    print(f"[INFO] Expected output: {expected_output}")

    try:
        agent_result = parse_int_list(*outer_args, **outer_kwargs)
        print(f"[INFO] Agent result: {agent_result}")
    except Exception as e:
        print(f"FAIL: Error executing parse_int_list with outer args: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("[INFO] Scenario B detected: Factory/Closure pattern")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from parse_int_list, got {type(agent_result)}")
            sys.exit(1)

        agent_operator = agent_result

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"[INFO] Inner args: {inner_args}")
            print(f"[INFO] Inner kwargs: {inner_kwargs}")
            print(f"[INFO] Inner expected output: {inner_expected}")

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Inner result: {result}")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner verification passed for: {inner_path}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected: Simple function")

        result = agent_result

        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Verification passed.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()