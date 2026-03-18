import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_full_propagate_to_sensor import full_propagate_to_sensor

# Import verification utility
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_ddnm_sandbox/run_code/std_data/data_full_propagate_to_sensor.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for full_propagate_to_sensor.")
        sys.exit(1)

    # --- Phase 1: Load outer data and run function ---
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Move tensors to appropriate device if needed
    def move_to_device(obj, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            moved = [move_to_device(x, device) for x in obj]
            return type(obj)(moved)
        if isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        return obj

    try:
        print("Executing full_propagate_to_sensor with outer args/kwargs...")
        agent_result = full_propagate_to_sensor(*outer_args, **outer_kwargs)
        print(f"Agent result type: {type(agent_result)}")
    except Exception as e:
        print(f"FAIL: Error executing full_propagate_to_sensor: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from full_propagate_to_sensor, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                print(f"\nLoading inner data [{idx}] from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Inner data keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                print(f"Executing agent_operator with inner args/kwargs [{idx}]...")
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data [{idx}]: {msg}")
                    all_passed = False
                else:
                    print(f"Inner data [{idx}] verification PASSED.")
            except Exception as e:
                print(f"FAIL: Error during verification for inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

        if not all_passed:
            sys.exit(1)
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function, comparing direct output.")

        expected = outer_output
        actual_result = agent_result

        try:
            passed, msg = recursive_check(expected, actual_result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()