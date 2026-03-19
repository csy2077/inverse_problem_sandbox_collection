import sys
import os
import logging

# Must patch logging into agent module before importing it
# The agent module uses logging.getLogger but doesn't import logging itself
import agent_compute_metric
agent_compute_metric.logging = logging

import dill
import torch
import numpy as np
import traceback

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_ADMMNet_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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
        print("ERROR: No outer data file found (data_compute_metric.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Debug: print info about args
    try:
        print(f"Number of outer args: {len(outer_args)}")
        for i, arg in enumerate(outer_args):
            if hasattr(arg, 'shape'):
                print(f"  arg[{i}]: type={type(arg).__name__}, shape={arg.shape}, dtype={getattr(arg, 'dtype', 'N/A')}")
            elif callable(arg):
                print(f"  arg[{i}]: callable, type={type(arg).__name__}")
            else:
                print(f"  arg[{i}]: type={type(arg).__name__}, value={arg}")
        print(f"Outer kwargs: {list(outer_kwargs.keys())}")
        print(f"Outer output type: {type(outer_output).__name__}")
    except Exception as e:
        print(f"Debug info failed: {e}")

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure Pattern ===")

        # Phase 1: Create operator
        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"Created agent_operator: type={type(agent_operator).__name__}")
        except Exception as e:
            print(f"ERROR: Failed to create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got type={type(agent_operator).__name__}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"\nLoaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution successful. Result type: {type(result).__name__}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"ERROR: Verification failed: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("\n=== Scenario A: Simple Function ===")

        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"Execution successful. Result type: {type(result).__name__}")
        except Exception as e:
            print(f"ERROR: Failed to execute compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Debug: print result and expected
        try:
            print(f"Expected: {expected}")
            print(f"Result:   {result}")
        except Exception:
            pass

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()