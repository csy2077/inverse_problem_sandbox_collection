import sys
import os
import logging
import dill
import torch
import numpy as np
import numpy
import traceback
import types

# Patch numpy as 'np' into builtins so any unpickled functions/closures can find it
import builtins
builtins.np = np

# Also ensure it's available in multiple places
sys.modules['np'] = np

# Pre-create the agent module with all necessary imports
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_compute_metric.py')

spec = types.ModuleType('agent_compute_metric')
spec.__file__ = agent_module_path
spec.__name__ = 'agent_compute_metric'

with open(agent_module_path, 'r') as f:
    source = f.read()

# Prepend necessary imports
patched_source = "import logging\nimport numpy as np\nimport numpy\nimport torch\n" + source

sys.modules['agent_compute_metric'] = spec
try:
    code = compile(patched_source, agent_module_path, 'exec')
    exec(code, spec.__dict__)
except Exception as e:
    print(f"FAIL: Could not load agent_compute_metric: {e}")
    traceback.print_exc()
    sys.exit(1)

from agent_compute_metric import compute_metric
from verification_utils import recursive_check


def patch_globals_recursive(obj, depth=0):
    """Patch np into the globals of any deserialized function/closure."""
    if depth > 10:
        return
    if callable(obj) and hasattr(obj, '__globals__'):
        obj.__globals__['np'] = np
        obj.__globals__['numpy'] = numpy
        obj.__globals__['torch'] = torch
    if hasattr(obj, '__func__') and hasattr(obj.__func__, '__globals__'):
        obj.__func__.__globals__['np'] = np
        obj.__func__.__globals__['numpy'] = numpy
        obj.__func__.__globals__['torch'] = torch
    if hasattr(obj, '__closure__') and obj.__closure__:
        for cell in obj.__closure__:
            try:
                cell_contents = cell.cell_contents
                if callable(cell_contents):
                    patch_globals_recursive(cell_contents, depth + 1)
            except (ValueError, AttributeError):
                pass


def patch_args(args):
    """Patch any callable in args to have np available."""
    patched = []
    for a in args:
        if callable(a):
            patch_globals_recursive(a)
        patched.append(a)
    return tuple(patched)


def patch_kwargs(kwargs):
    """Patch any callable in kwargs to have np available."""
    for k, v in kwargs.items():
        if callable(v):
            patch_globals_recursive(v)
    return kwargs


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_DS_DC1_sandbox/run_code/std_data/data_compute_metric.pkl'
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

    # Patch any callable arguments (like metric functions) to have np in their globals
    outer_args = patch_args(outer_args)
    outer_kwargs = patch_kwargs(outer_kwargs)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Created operator: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create operator from compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        patch_globals_recursive(agent_operator)

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

            inner_args = patch_args(inner_data.get('args', ()))
            inner_kwargs = patch_kwargs(inner_data.get('kwargs', {}))
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                    all_passed = False
                else:
                    print(f"  PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
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

        try:
            result = compute_metric(*outer_args, **outer_kwargs)
            print(f"  Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error executing compute_metric: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
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