import sys
import os
import logging

# Must patch logging before importing agent module
sys.modules.setdefault('logging', logging)

# Patch the agent module's globals before it tries to use logging
import importlib
import types

# Pre-create the module with logging available
agent_module_name = 'agent_compute_metric'
agent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_compute_metric.py')

# Read the source and prepend the import
with open(agent_path, 'r') as f:
    source = f.read()

# If logging is not imported in the source, prepend it
if 'import logging' not in source:
    source = 'import logging\n' + source

# Create module manually
agent_module = types.ModuleType(agent_module_name)
agent_module.__file__ = agent_path
agent_module.logging = logging
sys.modules[agent_module_name] = agent_module

exec(compile(source, agent_path, 'exec'), agent_module.__dict__)

import dill
import torch
import numpy as np
import traceback

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CNNAEU_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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
        print("FAIL: No outer data file found for compute_metric.")
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
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Created operator of type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                    all_passed = False
                else:
                    print(f"  PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Execute function
        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()