import sys
import os
import dill
import torch
import numpy as np
import traceback

# Determine data paths
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pigdm_sandbox/run_code/std_data/data_main.pkl']

# Separate outer (main) and inner (parent_function) paths
outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename or 'parent_' in basename:
        inner_paths.append(p)
    else:
        outer_path = p

print(f"Outer path: {outer_path}")
print(f"Inner paths: {inner_paths}")

# Determine scenario
is_factory = len(inner_paths) > 0

def load_data(path):
    """Load a pickle data file with dill."""
    print(f"Loading data from: {path}")
    with open(path, 'rb') as f:
        data = dill.load(f)
    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    return data

def main_test():
    from verification_utils import recursive_check

    # Phase 1: Load outer data
    if outer_path is None:
        print("ERROR: No outer data file found for main.")
        sys.exit(1)

    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else 'none'}")

    # Import main
    try:
        from agent_main import main
    except Exception as e:
        print(f"ERROR: Failed to import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    if is_factory:
        # Scenario B: Factory/Closure pattern
        print("\n=== SCENARIO B: Factory/Closure Pattern ===")

        # Phase 1: Create the operator
        try:
            print("Running main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create operator via main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"WARNING: agent_operator is not callable (type={type(agent_operator)}). Attempting direct comparison with outer output.")
            # Fallback: compare directly
            try:
                passed, msg = recursive_check(outer_output, agent_operator)
                if passed:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

        # Phase 2: Execute inner data
        for inner_path in inner_paths:
            try:
                inner_data = load_data(inner_path)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys()) if inner_kwargs else 'none'}")

            try:
                print("Running agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for inner data: {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for inner data: {os.path.basename(inner_path)}")
                    print(f"  Reason: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\n=== SCENARIO A: Simple Function ===")

        try:
            print("Running main(*outer_args, **outer_kwargs)...")
            result = main(*outer_args, **outer_kwargs)
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"TEST FAILED")
                print(f"  Reason: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main_test()