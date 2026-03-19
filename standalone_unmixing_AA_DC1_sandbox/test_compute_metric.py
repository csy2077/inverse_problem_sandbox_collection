import sys
import os
import logging
import dill
import torch
import numpy as np
import traceback

# Patch logging into agent_compute_metric before importing it
import importlib
import types

# First, ensure the agent module can access logging when it loads
# We need to inject logging into the module's namespace before it initializes
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_compute_metric.py')

# Read the source, prepend the import, and load it manually
with open(agent_module_path, 'r') as f:
    source = f.read()

if 'import logging' not in source:
    source = 'import logging\n' + source

module_name = 'agent_compute_metric'
spec = importlib.util.spec_from_loader(module_name, loader=None)
agent_module = types.ModuleType(module_name)
agent_module.__file__ = agent_module_path
agent_module.logging = logging
sys.modules[module_name] = agent_module

try:
    exec(compile(source, agent_module_path, 'exec'), agent_module.__dict__)
except Exception as e:
    print(f"FAIL: Could not load agent_compute_metric: {e}")
    traceback.print_exc()
    sys.exit(1)

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_AA_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Created agent_operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error running compute_metric (outer call): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from outer call, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Inner execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator (inner call): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Execution completed. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error running compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("  Verification PASSED")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()