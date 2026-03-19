import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
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
        print("FAIL: No outer data file (data_recon_slice.pkl) found.")
        sys.exit(1)

    # --- Phase 1: Load outer data and run recon_slice ---
    print(f"Loading outer data from: {outer_path}")
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

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Check if this is Scenario A or Scenario B
    if inner_paths:
        # --- Scenario B: Factory/Closure pattern ---
        print("Detected Scenario B (Factory/Closure pattern)")

        try:
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice (outer call) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from recon_slice, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Agent operator type: {type(agent_operator)}")

        # --- Phase 2: Load inner data and execute operator ---
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator (inner call) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # --- Comparison ---
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")

    else:
        # --- Scenario A: Simple function call ---
        print("Detected Scenario A (Simple function call)")

        try:
            result = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        if hasattr(expected, 'shape'):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")

        # --- Comparison ---
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)
        else:
            print("PASS: Output verified successfully.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()