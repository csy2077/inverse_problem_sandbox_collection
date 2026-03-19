import sys
import os
import dill
import traceback

# Import the target function
from agent_get_rect_coordinates_from_roi import get_rect_coordinates_from_roi

# Import verification utility
from verification_utils import recursive_check

def main():
    # Define data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_nnfbp_sandbox/run_code/std_data/data_get_rect_coordinates_from_roi.pkl'
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
        print("FAIL: Could not find outer data file (standard_data / data_get_rect_coordinates_from_roi.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct / execute
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Check if we have inner paths (Scenario B: Factory/Closure pattern)
    if inner_paths:
        # Scenario B: Factory pattern
        print("Detected Scenario B: Factory/Closure pattern.")

        # Step 1: Create the operator by calling the function with outer args
        try:
            agent_operator = get_rect_coordinates_from_roi(*outer_args, **outer_kwargs)
            print("  Successfully created agent_operator from outer data.")
        except Exception as e:
            print(f"FAIL: Error creating agent_operator with outer args/kwargs.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify operator is callable
        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable. Got type: {type(agent_operator)}")
            sys.exit(1)

        # Step 2: For each inner data file, execute and verify
        all_passed = True
        for ip in inner_paths:
            try:
                with open(ip, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data from: {ip}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {ip}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Successfully executed agent_operator with inner args.")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args/kwargs.")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data: {ip}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"  Verification passed for inner data: {os.path.basename(ip)}")
            except Exception as e:
                print(f"FAIL: Error during recursive_check for inner data: {ip}")
                print(f"  Error: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED: One or more inner verifications failed.")
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call.")

        try:
            result = get_rect_coordinates_from_roi(*outer_args, **outer_kwargs)
            print("  Successfully executed get_rect_coordinates_from_roi with outer args.")
        except Exception as e:
            print(f"FAIL: Error executing get_rect_coordinates_from_roi with outer args/kwargs.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Error during recursive_check.")
            print(f"  Error: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()