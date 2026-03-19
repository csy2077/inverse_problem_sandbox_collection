import sys
import os
import logging
import dill
import torch
import numpy as np
import traceback

# Fix the agent module's missing logging import before importing from it
import importlib
import types

# Pre-load the module source and inject logging
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_compute_metric.py')
with open(agent_module_path, 'r') as f:
    source = f.read()

# If logging is not imported in the agent module, prepend it
if 'import logging' not in source:
    source = 'import logging\n' + source

# Create the module manually
spec = importlib.util.spec_from_file_location("agent_compute_metric", agent_module_path)
agent_module = types.ModuleType("agent_compute_metric")
agent_module.__file__ = agent_module_path
agent_module.logging = logging
sys.modules['agent_compute_metric'] = agent_module
exec(compile(source, agent_module_path, 'exec'), agent_module.__dict__)

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR creating agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: agent_operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR loading inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Execution successful, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED for inner data {os.path.basename(inner_path)}:")
                    print(f"  {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Execution successful, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR executing compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED:")
                print(f"  {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()