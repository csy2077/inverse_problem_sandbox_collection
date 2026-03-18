import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_load_config import load_config
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_load_config.pkl'
    ]

    # Step 1: Classify paths into outer (direct) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for load_config.")
        sys.exit(1)

    # Step 2: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Step 3: Execute load_config with outer args/kwargs
    try:
        agent_operator = load_config(*outer_args, **outer_kwargs)
        print(f"load_config executed successfully. Result type: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: load_config raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        # Verify agent_operator is callable
        if not callable(agent_operator):
            print(f"FAIL: Expected callable from load_config, got {type(agent_operator)}")
            sys.exit(1)

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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Inner execution succeeded. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Inner execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner check passed for: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - result from Phase 1 IS the result
        print("Scenario A detected: No inner data files. Comparing direct output.")
        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("  Direct output check passed.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()