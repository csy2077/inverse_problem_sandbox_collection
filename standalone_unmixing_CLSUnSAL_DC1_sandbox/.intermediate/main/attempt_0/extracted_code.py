import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # --- Scenario A: Simple Function (no inner paths) ---
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Outer data keys: {list(outer_data.keys())}")
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Import main function
    try:
        from agent_main import main as target_main
        print("Successfully imported 'main' from agent_main.")
    except Exception as e:
        print(f"ERROR importing main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        # --- Scenario B: Factory/Closure Pattern ---
        print("Detected Scenario B: Factory/Closure Pattern")

        # Phase 1: Reconstruct operator
        try:
            print("Phase 1: Calling main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = target_main(*outer_args, **outer_kwargs)
            print(f"Operator returned: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR during Phase 1 (operator creation): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Phase 2: Executing operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR during Phase 2 (operator execution): {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # --- Scenario A: Simple Function ---
        print("Detected Scenario A: Simple Function")

        # Phase 1: Call main
        try:
            print("Calling main(*outer_args, **outer_kwargs)...")
            result = target_main(*outer_args, **outer_kwargs)
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR during main execution: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nAll tests passed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()