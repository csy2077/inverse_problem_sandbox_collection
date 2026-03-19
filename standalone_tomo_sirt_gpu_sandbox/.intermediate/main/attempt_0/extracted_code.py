import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main_test():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (main) and inner (parent_function) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    # --- Phase 1: Load outer data and run main ---
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Import main from agent_main
    try:
        from agent_main import main
    except ImportError as e:
        print(f"ERROR: Failed to import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Execute main
    try:
        print("Executing main(*args, **kwargs)...")
        actual_result = main(*outer_args, **outer_kwargs)
        print("main() execution completed.")
    except Exception as e:
        print(f"ERROR: main() execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify that main returned a callable
        if not callable(actual_result):
            print(f"ERROR: main() did not return a callable. Got type: {type(actual_result)}")
            sys.exit(1)

        agent_operator = actual_result
        all_passed = True

        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)} ---")

            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Executing agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Inner execution completed.")
            except Exception as e:
                print(f"ERROR: agent_operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED inner test {idx + 1}: {msg}")
                all_passed = False
            else:
                print(f"Inner test {idx + 1} PASSED.")

        if not all_passed:
            print("\nTEST FAILED: One or more inner tests did not pass.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function, comparing output directly.")

        result = actual_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"ERROR: recursive_check failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main_test()