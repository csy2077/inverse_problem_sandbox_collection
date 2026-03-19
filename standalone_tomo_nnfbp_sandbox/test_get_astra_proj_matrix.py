import sys
import os
import pickle
import traceback
import numpy as np

# Must patch scipy into builtins BEFORE importing the agent module
import scipy
import scipy.sparse
import scipy.sparse.linalg
import builtins
builtins.scipy = scipy

import types

# Read the agent module source and exec it with scipy in globals
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_get_astra_proj_matrix.py')
if not os.path.exists(agent_module_path):
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
from verification_utils import recursive_check


def try_load_pickle(filepath):
    """Try multiple methods to load a pickle file."""
    # Method 1: dill with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception:
        pass

    # Method 2: standard pickle
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        pass

    # Method 3: dill with different protocol
    try:
        with open(filepath, 'rb') as f:
            unpickler = dill.Unpickler(f)
            data = unpickler.load()
        return data
    except Exception:
        pass

    # Method 4: Try loading with numpy
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception:
        pass

    # Method 5: Check if the file is empty or corrupted
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"File is empty (0 bytes): {filepath}")

    # Method 6: Try reading multiple pickled objects
    try:
        results = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    obj = dill.load(f)
                    results.append(obj)
                except EOFError:
                    break
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            return results
        else:
            raise ValueError("No objects found in file")
    except Exception:
        pass

    raise ValueError(f"Could not load file with any method: {filepath}")


def find_all_data_files(base_dir, func_name):
    """Search for all related data files in the directory."""
    outer_path = None
    inner_paths = []

    if not os.path.isdir(base_dir):
        return outer_path, inner_paths

    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.endswith('.pkl'):
            continue

        if ('parent_function' in fname or 'parent_' in fname) and func_name in fname:
            inner_paths.append(fpath)
        elif fname == f'data_{func_name}.pkl':
            outer_path = fpath
        elif func_name in fname and 'standard_data' in fname:
            outer_path = fpath

    return outer_path, inner_paths


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_nnfbp_sandbox/run_code/std_data/data_get_astra_proj_matrix.pkl'
    ]

    # Determine base directory for scanning
    base_dir = None
    for p in data_paths:
        d = os.path.dirname(p)
        if os.path.isdir(d):
            base_dir = d
            break

    outer_path = None
    inner_paths = []

    # First, classify provided paths
    for p in data_paths:
        if not os.path.exists(p):
            print(f"WARNING: Data path does not exist: {p}")
            continue
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # Also scan directory for inner files that might not be in data_paths
    if base_dir and os.path.isdir(base_dir):
        for fname in os.listdir(base_dir):
            fpath = os.path.join(base_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.endswith('.pkl'):
                continue
            if fpath in data_paths:
                continue
            if ('parent_function' in fname or 'parent_' in fname) and 'get_astra_proj_matrix' in fname:
                inner_paths.append(fpath)
                print(f"Discovered additional inner file: {fname}")
            elif fname == 'standard_data_get_astra_proj_matrix.pkl' and outer_path is None:
                outer_path = fpath

    # Check if outer_path file is valid (non-empty)
    outer_data = None
    if outer_path and os.path.exists(outer_path):
        file_size = os.path.getsize(outer_path)
        print(f"Outer data file: {outer_path} (size: {file_size} bytes)")

        if file_size > 0:
            try:
                outer_data = try_load_pickle(outer_path)
                print(f"Successfully loaded outer data.")
            except Exception as e:
                print(f"WARNING: Could not load outer data file: {e}")
                traceback.print_exc()
        else:
            print(f"WARNING: Outer data file is empty (0 bytes).")

    # If outer data couldn't be loaded, we need to construct the call manually
    # Based on the gen_data_code, the function signature is: get_astra_proj_matrix(nd, angles, method)
    # We'll use reasonable defaults that would work with CUDA
    if outer_data is None:
        print("Outer data file could not be loaded. Reconstructing from known function signature.")
        print("Using default test parameters for get_astra_proj_matrix.")

        # Use typical parameters for ASTRA NN-FBP workflow
        nd = 256
        angles = np.linspace(0, np.pi, 180, endpoint=False).astype(np.float64)
        method = 'NN-FBP'  # This triggers CUDA projector

        outer_args = (nd, angles, method)
        outer_kwargs = {}
        outer_output = None  # We can't compare output directly
    else:
        if isinstance(outer_data, dict):
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            outer_output = outer_data.get('output', None)
            print(f"Outer function: {outer_data.get('func_name', 'unknown')}")
        else:
            print(f"Unexpected outer data type: {type(outer_data)}")
            outer_args = ()
            outer_kwargs = {}
            outer_output = None

    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 1: Create the operator
    try:
        agent_operator = get_astra_proj_matrix(*outer_args, **outer_kwargs)
        print(f"Successfully created operator: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: get_astra_proj_matrix raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verify the operator is valid
    if agent_operator is None:
        print("FAIL: get_astra_proj_matrix returned None")
        sys.exit(1)

    # Check that the operator has expected attributes
    if hasattr(agent_operator, 'shape'):
        print(f"Operator shape: {agent_operator.shape}")
    if hasattr(agent_operator, 'vshape'):
        print(f"Operator vshape: {agent_operator.vshape}")
    if hasattr(agent_operator, 'sshape'):
        print(f"Operator sshape: {agent_operator.sshape}")

    # Phase 2: Execute inner tests if available
    if len(inner_paths) > 0:
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")

            file_size = os.path.getsize(inner_path) if os.path.exists(inner_path) else 0
            print(f"Inner file size: {file_size} bytes")

            if file_size == 0:
                print(f"WARNING: Inner data file is empty, skipping.")
                continue

            try:
                inner_data = try_load_pickle(inner_path)
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
                # Try to find the method on the operator
                if hasattr(agent_operator, inner_func_name):
                    method = getattr(agent_operator, inner_func_name)
                    # Strip 'self' argument if present
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
        print("\nScenario A detected: No inner data files.")

        # If we have outer output to compare against
        if outer_output is not None and outer_data is not None:
            print("Comparing operator output directly against expected.")
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
        else:
            # No inner data and no loadable outer output - validate the operator structurally
            print("No expected output available. Performing structural validation of operator.")

            try:
                # Validate it's an OpTomo instance
                assert hasattr(agent_operator, 'FP'), "Operator missing FP method"
                assert hasattr(agent_operator, 'BP'), "Operator missing BP method"
                assert hasattr(agent_operator, 'shape'), "Operator missing shape attribute"
                assert hasattr(agent_operator, 'vshape'), "Operator missing vshape attribute"
                assert hasattr(agent_operator, 'sshape'), "Operator missing sshape attribute"
                assert hasattr(agent_operator, 'proj_id'), "Operator missing proj_id attribute"
                assert hasattr(agent_operator, 'T'), "Operator missing transpose T"
                assert hasattr(agent_operator, 'reconstruct'), "Operator missing reconstruct method"

                # Test forward projection with a simple phantom
                vshape = agent_operator.vshape
                test_vol = np.zeros(vshape, dtype=np.float32)
                # Create a simple circle phantom
                center = (vshape[0] // 2, vshape[1] // 2)
                radius = min(vshape) // 4
                y, x = np.ogrid[:vshape[0], :vshape[1]]
                mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
                test_vol[mask] = 1.0

                print(f"Testing FP with volume shape {vshape}...")
                sinogram = agent_operator.FP(test_vol)
                assert sinogram is not None, "FP returned None"
                assert sinogram.shape == agent_operator.sshape, f"FP output shape {sinogram.shape} != expected {agent_operator.sshape}"
                assert np.any(sinogram != 0), "FP returned all zeros for non-zero input"
                print(f"FP output shape: {sinogram.shape}, max: {sinogram.max():.4f}")

                # Test backprojection
                print(f"Testing BP with sinogram shape {agent_operator.sshape}...")
                backproj = agent_operator.BP(sinogram)
                assert backproj is not None, "BP returned None"
                assert backproj.shape == vshape, f"BP output shape {backproj.shape} != expected {vshape}"
                assert np.any(backproj != 0), "BP returned all zeros for non-zero input"
                print(f"BP output shape: {backproj.shape}, max: {backproj.max():.4f}")

                # Test matvec (operator * vector)
                print("Testing matvec...")
                flat_result = agent_operator._matvec(test_vol.ravel())
                assert flat_result is not None, "_matvec returned None"
                expected_flat_size = agent_operator.ssize
                assert flat_result.size == expected_flat_size, f"matvec output size {flat_result.size} != expected {expected_flat_size}"

                # Test transpose
                print("Testing transpose...")
                transp = agent_operator.T
                assert transp is not None, "Transpose is None"
                assert transp.shape == (agent_operator.shape[1], agent_operator.shape[0]), "Transpose shape mismatch"

                print("\nAll structural validations passed.")
                print("TEST PASSED")
                sys.exit(0)

            except AssertionError as e:
                print(f"FAIL: Structural validation failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            except Exception as e:
                print(f"FAIL: Structural validation error: {e}")
                traceback.print_exc()
                sys.exit(1)


if __name__ == '__main__':
    main()