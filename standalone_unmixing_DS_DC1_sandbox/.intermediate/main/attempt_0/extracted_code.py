import sys
import os
import dill
import numpy as np
import traceback

# Attempt to import the target function
try:
    from agent_main import main
except Exception as e:
    print(f"FATAL: Could not import 'main' from 'agent_main': {e}")
    traceback.print_exc()
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except Exception as e:
    print(f"FATAL: Could not import 'recursive_check' from 'verification_utils': {e}")
    traceback.print_exc()
    sys.exit(1)


def test_main():
    """Test the main function using captured data."""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_DS_DC1_sandbox/run_code/std_data/data_main.pkl'
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
        print("FATAL: No outer data file (data_main.pkl) found in data_paths.")
        sys.exit(1)

    # -------------------------------------------------------
    # Phase 1: Load outer data and run main()
    # -------------------------------------------------------
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FATAL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, Outer kwargs keys: {list(outer_kwargs.keys())}")

    # -------------------------------------------------------
    # Determine scenario
    # -------------------------------------------------------
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        print(f"Found {len(inner_paths)} inner data file(s).")

        # Run main to get the operator/closure
        try:
            print("Running main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FATAL: main() raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"WARNING: agent_operator is not callable (type={type(agent_operator)}). "
                  f"Attempting to use it as result directly.")

        # Process each inner path
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FATAL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner args count: {len(inner_args)}, Inner kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Running agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FATAL: agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FATAL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED inner test {idx + 1}: {msg}")
                all_passed = False
            else:
                print(f"PASSED inner test {idx + 1}")

        if not all_passed:
            print("\nTEST FAILED: One or more inner tests did not pass.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            print("Running main(*outer_args, **outer_kwargs)...")
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FATAL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FATAL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == "__main__":
    test_main()