import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import target function
from agent_weight_init import weight_init
from verification_utils import recursive_check


def main():
    # All data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_pnpdm_sandbox/run_code/std_data/data_weight_init.pkl'
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
        print("FAIL: No outer data file found for weight_init.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Run outer function to get operator
        try:
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            print(f"  Created operator of type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from weight_init: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"  PASSED for {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        expected = outer_data.get('output')

        try:
            # Set random seeds to match the recorded output if possible
            # Note: weight_init uses torch.rand/torch.randn which are stochastic.
            # The recorded data should have been captured with deterministic state,
            # but we compare shapes/dtypes via recursive_check which may handle tolerance.
            result = weight_init(*outer_args, **outer_kwargs)
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute weight_init: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()