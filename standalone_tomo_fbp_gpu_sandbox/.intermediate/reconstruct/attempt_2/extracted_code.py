import sys
import os
import traceback

# Must patch scipy into builtins/globals before importing agent_reconstruct
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Inject scipy into the module's expected global namespace
import builtins
builtins.scipy = scipy

# Also ensure it's in sys.modules
import importlib
sys.modules['scipy'] = scipy
sys.modules['scipy.sparse'] = scipy.sparse
sys.modules['scipy.sparse.linalg'] = scipy.sparse.linalg

# Now patch the agent_reconstruct module's globals before it tries to use scipy
# We need to load it manually to inject scipy
import types

agent_module_path = None
for search_dir in sys.path + [os.getcwd()]:
    candidate = os.path.join(search_dir, 'agent_reconstruct.py')
    if os.path.exists(candidate):
        agent_module_path = candidate
        break

if agent_module_path is None:
    # Try current directory
    if os.path.exists('agent_reconstruct.py'):
        agent_module_path = 'agent_reconstruct.py'
    else:
        print("ERROR: Cannot find agent_reconstruct.py")
        sys.exit(1)

# Read the source and inject scipy import at the top
with open(agent_module_path, 'r') as f:
    source = f.read()

# Create the module manually with scipy pre-injected
agent_module = types.ModuleType('agent_reconstruct')
agent_module.__file__ = agent_module_path
agent_module.scipy = scipy

# Add scipy to the module's namespace before exec
exec_globals = agent_module.__dict__
exec_globals['scipy'] = scipy

# If the source doesn't import scipy, we prepend it
if 'import scipy' not in source:
    source = "import scipy\nimport scipy.sparse\nimport scipy.sparse.linalg\n" + source

try:
    exec(compile(source, agent_module_path, 'exec'), exec_globals)
except Exception as e:
    print(f"ERROR: Failed to load agent_reconstruct: {e}")
    traceback.print_exc()
    sys.exit(1)

sys.modules['agent_reconstruct'] = agent_module
reconstruct = agent_module.reconstruct

import dill
import numpy as np
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_reconstruct.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file (data_reconstruct.pkl) found.")
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

        # Phase 1: Create the operator
        try:
            agent_operator = reconstruct(*outer_args, **outer_kwargs)
            print(f"  Created operator of type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create operator via reconstruct(): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify operator is callable
        if not callable(agent_operator):
            print(f"ERROR: Returned operator is not callable. Type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
                print(f"  func_name: {inner_data.get('func_name', 'N/A')}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Operator execution succeeded. Result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
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
                print(f"ERROR: Verification failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Execute the function
        try:
            result = reconstruct(*outer_args, **outer_kwargs)
            print(f"  Function execution succeeded. Result type: {type(result)}")
            if isinstance(result, np.ndarray):
                print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
        except Exception as e:
            print(f"ERROR: Failed to execute reconstruct(): {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

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
            print(f"ERROR: Verification failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()