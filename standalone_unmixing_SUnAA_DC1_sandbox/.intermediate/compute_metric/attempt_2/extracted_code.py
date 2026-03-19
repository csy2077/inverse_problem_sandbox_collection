import sys
import os
import logging
import dill
import torch
import numpy as np
import traceback

# Patch logging into agent module before importing compute_metric
import agent_compute_metric
agent_compute_metric.logging = logging

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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
        print("ERROR: No outer data file found for compute_metric.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"Phase 1: compute_metric returned: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to call compute_metric (outer): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable from compute_metric, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: agent_operator returned: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"VERIFICATION FAILED for {inner_path}: {msg}")
                    all_passed = False
                else:
                    print(f"Verification passed for {inner_path}")
            except Exception as e:
                print(f"ERROR: Verification exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"compute_metric returned: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to call compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"VERIFICATION FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Verification exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()