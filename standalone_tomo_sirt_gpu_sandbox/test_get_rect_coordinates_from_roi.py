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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_get_rect_coordinates_from_roi.pkl'
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

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer args: {outer_args}")
    print(f"Outer kwargs: {outer_kwargs}")

    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = get_rect_coordinates_from_roi(*outer_args, **outer_kwargs)
            print(f"Agent operator created: {agent_operator}")
        except Exception as e:
            print(f"FAIL: Could not create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Agent operator is not callable. Got type: {type(agent_operator)}")
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

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"Inner args types: {[type(a).__name__ for a in inner_args]}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Execution of agent operator failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")

        try:
            result = get_rect_coordinates_from_roi(*outer_args, **outer_kwargs)
            print(f"Function executed. Result type: {type(result)}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Comparison
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                print(f"Expected: {expected_output}")
                print(f"Got: {result}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()