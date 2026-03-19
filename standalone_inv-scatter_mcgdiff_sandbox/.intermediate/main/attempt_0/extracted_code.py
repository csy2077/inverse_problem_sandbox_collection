import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_main import main
from verification_utils import recursive_check


def load_pkl(path):
    """Load a pickle file using dill."""
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def test_main():
    """Test the main function against captured standard data."""
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_mcgdiff_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Separate outer (direct main) and inner (parent_function/closure) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, f"Could not find outer data file (data_main.pkl) in {data_paths}"

    # ---- Phase 1: Load outer data and run main ----
    print(f"Loading outer data from: {outer_path}")
    try:
        outer_data = load_pkl(outer_path)
    except Exception as e:
        print(f"FAILED: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if inner_paths:
        # ---- Scenario B: Factory/Closure Pattern ----
        print("Detected Scenario B (Factory/Closure pattern)")
        print(f"Found {len(inner_paths)} inner data file(s)")

        # Run main to get the operator/closure
        print("Running main(*args, **kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        assert callable(agent_operator), (
            f"FAILED: Expected main() to return a callable operator, got {type(agent_operator)}"
        )
        print(f"Got callable operator: {type(agent_operator)}")

        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                inner_data = load_pkl(inner_path)
            except Exception as e:
                print(f"FAILED: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            # Execute the operator with inner args
            print("Running agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED: agent_operator() raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            print("Comparing actual result with expected output...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"FAILED: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED: Result mismatch for inner data {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for inner data: {os.path.basename(inner_path)}")

    else:
        # ---- Scenario A: Simple Function ----
        print("Detected Scenario A (Simple function call)")

        print("Running main(*args, **kwargs)...")
        try:
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare results
        print("Comparing actual result with expected output...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"FAILED: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAILED: Result mismatch")
            print(f"Message: {msg}")
            sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    test_main()