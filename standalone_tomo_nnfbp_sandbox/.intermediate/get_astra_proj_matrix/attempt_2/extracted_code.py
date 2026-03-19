import sys
import os

# Must patch scipy into builtins BEFORE importing the agent module
import scipy
import scipy.sparse
import scipy.sparse.linalg
import builtins
builtins.scipy = scipy

# Now patch it into the module's global namespace by pre-loading it
import importlib
import types

# Read the agent module source and exec it with scipy in globals
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_get_astra_proj_matrix.py')
if not os.path.exists(agent_module_path):
    # Try current directory
    agent_module_path = 'agent_get_astra_proj_matrix.py'

agent_module = types.ModuleType('agent_get_astra_proj_matrix')
agent_module.__file__ = agent_module_path
agent_module.scipy = scipy

with open(agent_module_path, 'r') as f:
    source = f.read()

exec(compile(source, agent_module_path, 'exec'), agent_module.__dict__)
sys.modules['agent_get_astra_proj_matrix'] = agent_module

from agent_get_astra_proj_matrix import get_astra_proj_matrix

import dill
import numpy as np
import traceback
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_nnfbp_sandbox/run_code/std_data/data_get_astra_proj_matrix.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for get_astra_proj_matrix.")
        sys.exit(1)

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

    print(f"Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_operator = get_astra_proj_matrix(*outer_args, **outer_kwargs)
        print(f"Successfully created operator: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: get_astra_proj_matrix raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    if len(inner_paths) > 0:
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                all_passed = False
                continue

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            inner_func_name = inner_data.get('func_name', 'unknown')

            print(f"Inner function: {inner_func_name}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                if hasattr(agent_operator, inner_func_name):
                    method = getattr(agent_operator, inner_func_name)
                    if len(inner_args) > 0:
                        first_arg = inner_args[0]
                        if hasattr(first_arg, '__class__') and first_arg.__class__.__name__ == agent_operator.__class__.__name__:
                            inner_args = inner_args[1:]
                    actual_result = method(*inner_args, **inner_kwargs)
                else:
                    actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Execution of inner function raised an exception: {e}")
                traceback.print_exc()
                all_passed = False
                continue

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                all_passed = False
                continue

            if not passed:
                print(f"FAIL: Inner test {idx + 1} failed: {msg}")
                all_passed = False
            else:
                print(f"PASS: Inner test {idx + 1} passed.")

        if not all_passed:
            print("\nTEST FAILED: One or more inner tests failed.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        print("\nScenario A detected: No inner data files. Comparing operator output directly.")

        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()