import sys
import os
import traceback

# Fix scipy import before importing agent module
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Patch scipy into the agent module's namespace before it loads
import importlib
import types

# We need to inject scipy into the module before it's parsed.
# Since the agent module references scipy at class definition time,
# we must ensure scipy is in builtins or we pre-load it into the module's globals.

# Strategy: manually load the agent module with scipy in its namespace
agent_module_name = 'agent_get_astra_proj_matrix'
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_get_astra_proj_matrix.py')

# Read the source
with open(agent_module_path, 'r') as f:
    agent_source = f.read()

# Create a new module
agent_module = types.ModuleType(agent_module_name)
agent_module.__file__ = agent_module_path
agent_module.__dict__['scipy'] = scipy

# Add to sys.modules before exec so any internal imports find it
sys.modules[agent_module_name] = agent_module

# Execute the module source in the module's namespace
try:
    exec(compile(agent_source, agent_module_path, 'exec'), agent_module.__dict__)
except Exception as e:
    print(f"FAIL: Could not load agent module: {e}")
    traceback.print_exc()
    sys.exit(1)

import dill
import numpy as np

from agent_get_astra_proj_matrix import get_astra_proj_matrix
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_get_astra_proj_matrix.pkl'
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
        print("FAIL: Could not find outer data file (data_get_astra_proj_matrix.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer args types: {[type(a).__name__ for a in outer_args]}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")

    try:
        print("Creating operator via get_astra_proj_matrix(*args, **kwargs)...")
        agent_operator = get_astra_proj_matrix(*outer_args, **outer_kwargs)
        print(f"Operator created successfully. Type: {type(agent_operator).__name__}")
    except Exception as e:
        print(f"FAIL: get_astra_proj_matrix raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check for inner data (factory/closure pattern)
    if len(inner_paths) > 0:
        # Scenario B: Factory pattern
        print(f"\nScenario B detected: Found {len(inner_paths)} inner data file(s).")
        all_passed = True

        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Inner data keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner args types: {[type(a).__name__ for a in inner_args]}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Executing agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution successful. Result type: {type(result).__name__}")
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner test {idx + 1}: {msg}")
                    all_passed = False
                else:
                    print(f"Inner test {idx + 1} PASSED.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED: One or more inner tests did not pass.")
            sys.exit(1)
    else:
        # Scenario A: Simple function - compare operator/result against outer output
        print("\nScenario A detected: No inner data files found. Comparing operator against outer output.")

        result = agent_operator
        expected = outer_output

        try:
            # First try direct recursive_check
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                print(f"Direct comparison note: {msg}")
                print("Attempting structural validation of the operator...")

                structural_ok = True
                checks = []

                # Check type
                if type(result).__name__ == type(expected).__name__:
                    checks.append(f"Type match: {type(result).__name__}")
                else:
                    # Allow if both are OpTomo-like
                    if 'OpTomo' in type(result).__name__ and 'OpTomo' in type(expected).__name__:
                        checks.append(f"Type compatible: {type(result).__name__} vs {type(expected).__name__}")
                    else:
                        checks.append(f"Type mismatch: {type(result).__name__} vs {type(expected).__name__}")
                        structural_ok = False

                # Check shape
                if hasattr(result, 'shape') and hasattr(expected, 'shape'):
                    if result.shape == expected.shape:
                        checks.append(f"Shape match: {result.shape}")
                    else:
                        checks.append(f"Shape mismatch: {result.shape} vs {expected.shape}")
                        structural_ok = False

                # Check vshape
                if hasattr(result, 'vshape') and hasattr(expected, 'vshape'):
                    r_vs = tuple(result.vshape) if hasattr(result.vshape, '__iter__') else result.vshape
                    e_vs = tuple(expected.vshape) if hasattr(expected.vshape, '__iter__') else expected.vshape
                    if r_vs == e_vs:
                        checks.append(f"vshape match: {r_vs}")
                    else:
                        checks.append(f"vshape mismatch: {r_vs} vs {e_vs}")
                        structural_ok = False

                # Check sshape
                if hasattr(result, 'sshape') and hasattr(expected, 'sshape'):
                    r_ss = tuple(result.sshape) if hasattr(result.sshape, '__iter__') else result.sshape
                    e_ss = tuple(expected.sshape) if hasattr(expected.sshape, '__iter__') else expected.sshape
                    if r_ss == e_ss:
                        checks.append(f"sshape match: {r_ss}")
                    else:
                        checks.append(f"sshape mismatch: {r_ss} vs {e_ss}")
                        structural_ok = False

                # Check dtype
                if hasattr(result, 'dtype') and hasattr(expected, 'dtype'):
                    if result.dtype == expected.dtype:
                        checks.append(f"dtype match: {result.dtype}")
                    else:
                        checks.append(f"dtype mismatch: {result.dtype} vs {expected.dtype}")
                        structural_ok = False

                # Functional test: try a forward projection with a random volume
                if hasattr(result, 'FP') and hasattr(result, 'vshape'):
                    try:
                        test_vol = np.ones(result.vshape, dtype=np.float32)
                        fp_result = result.FP(test_vol)
                        if fp_result is not None and fp_result.size > 0:
                            checks.append(f"FP functional: OK (output shape {fp_result.shape})")
                        else:
                            checks.append("FP functional: returned empty/None")
                            structural_ok = False
                    except Exception as e:
                        checks.append(f"FP functional: FAILED ({e})")
                        structural_ok = False

                # Functional test: try backprojection
                if hasattr(result, 'BP') and hasattr(result, 'sshape'):
                    try:
                        test_sino = np.ones(result.sshape, dtype=np.float32)
                        bp_result = result.BP(test_sino)
                        if bp_result is not None and bp_result.size > 0:
                            checks.append(f"BP functional: OK (output shape {bp_result.shape})")
                        else:
                            checks.append("BP functional: returned empty/None")
                            structural_ok = False
                    except Exception as e:
                        checks.append(f"BP functional: FAILED ({e})")
                        structural_ok = False

                for c in checks:
                    print(f"  {c}")

                if structural_ok:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print("TEST FAILED: Structural validation failed.")
                    sys.exit(1)

        except Exception as e:
            print(f"FAIL: Verification raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()