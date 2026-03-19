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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_lgd_sandbox/run_code/std_data/data_full_propagate_to_sensor.pkl'
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
        print("FATAL: Could not find outer data file for full_propagate_to_sensor.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct/run function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
        print(f"[INFO] func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FATAL: Failed to load outer data: {e}")
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
        # Execute the function with outer args
        print("[INFO] Running full_propagate_to_sensor with outer args...")
        agent_result = full_propagate_to_sensor(*outer_args, **outer_kwargs)
        print(f"[INFO] Function returned type: {type(agent_result)}")
    except Exception as e:
        print(f"FATAL: Failed to execute full_propagate_to_sensor: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected. Found {len(inner_paths)} inner data file(s).")

        if not callable(agent_result):
            print(f"FATAL: Expected callable from full_propagate_to_sensor, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FATAL: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                print("[INFO] Executing agent_result (closure) with inner args...")
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FATAL: Failed to execute closure: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"[INFO] Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FATAL: Verification raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)

    else:
        # Scenario A: Simple function
        print("[INFO] Scenario A detected. Comparing direct output.")
        expected = outer_output
        result = agent_result

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
            print(f"FATAL: Verification raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()