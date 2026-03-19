import sys
import os
import dill
import numpy as np
import traceback
import pickle

# Ensure the current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def try_load_pickle(filepath):
    """Try multiple methods to load a pickle file."""
    errors = []

    # Method 1: dill with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"dill.load: {e}")

    # Method 2: pickle with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        errors.append(f"pickle.load: {e}")

    # Method 3: Try loading multiple objects (sometimes files have multiple dumps)
    try:
        objects = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    obj = dill.load(f)
                    objects.append(obj)
                except EOFError:
                    break
        if len(objects) == 1:
            return objects[0]
        elif len(objects) > 1:
            # Try to reconstruct as a dict if we got multiple objects
            if isinstance(objects[0], str):
                # Might be key-value pairs
                result = {}
                for i in range(0, len(objects) - 1, 2):
                    result[objects[i]] = objects[i + 1]
                return result
            return objects
    except Exception as e:
        errors.append(f"multi-object dill: {e}")

    # Method 4: Try with different pickle protocols
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        errors.append(f"pickle latin1: {e}")

    # Method 5: Check if file is actually a different format or empty
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise RuntimeError(f"File is empty (0 bytes): {filepath}")

    # Method 6: Try reading with dill Unpickler with different settings
    try:
        with open(filepath, 'rb') as f:
            unpickler = dill.Unpickler(f)
            data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"dill.Unpickler: {e}")

    raise RuntimeError(
        f"Could not load {filepath} (size={file_size} bytes). Errors:\n" +
        "\n".join(f"  - {err}" for err in errors)
    )


def find_pkl_files(base_dir, func_name):
    """Search for pickle files related to the function in the directory."""
    results = []
    if not os.path.isdir(base_dir):
        return results
    for fname in os.listdir(base_dir):
        if fname.endswith('.pkl') and func_name in fname:
            results.append(os.path.join(base_dir, fname))
    return results


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Also search the std_data directory for any related files we might have missed
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        all_pkl_files = find_pkl_files(std_data_dir, 'recon_slice')
        for p in all_pkl_files:
            if p not in data_paths:
                data_paths.append(p)
        print(f"All discovered pkl files for recon_slice: {[os.path.basename(p) for p in data_paths]}")

    # Check file sizes
    for p in data_paths:
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  {os.path.basename(p)}: {size} bytes")
        else:
            print(f"  {os.path.basename(p)}: FILE NOT FOUND")

    # Classify paths into outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if not os.path.exists(p):
            print(f"WARNING: File not found, skipping: {p}")
            continue
        if os.path.getsize(p) == 0:
            print(f"WARNING: File is empty, skipping: {p}")
            continue
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # If the primary file is missing or empty, try to find an alternative
    if outer_path is None:
        # Try standard_data variant
        alt_path = os.path.join(std_data_dir, 'standard_data_recon_slice.pkl')
        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
            outer_path = alt_path
            print(f"Using alternative outer path: {alt_path}")

    if outer_path is None:
        print("FAIL: No valid outer data file found.")
        sys.exit(1)

    # --- Phase 1: Load outer data ---
    try:
        print(f"\nLoading outer data from: {outer_path}")
        outer_data = try_load_pickle(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Handle different data formats
    if isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    elif isinstance(outer_data, (list, tuple)):
        # Maybe it's stored as (args, kwargs, output) or similar
        if len(outer_data) >= 3:
            outer_args = outer_data[0] if isinstance(outer_data[0], (list, tuple)) else (outer_data[0],)
            outer_kwargs = outer_data[1] if isinstance(outer_data[1], dict) else {}
            outer_output = outer_data[2]
        else:
            print(f"FAIL: Unexpected data format (list/tuple of length {len(outer_data)})")
            sys.exit(1)
    else:
        print(f"FAIL: Unexpected data format: {type(outer_data)}")
        print(f"Data preview: {str(outer_data)[:500]}")
        sys.exit(1)

    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Print info about args for debugging
    for i, arg in enumerate(outer_args):
        if hasattr(arg, 'shape'):
            print(f"  arg[{i}]: {type(arg).__name__}, shape={arg.shape}, dtype={getattr(arg, 'dtype', 'N/A')}")
        elif hasattr(arg, '__len__'):
            print(f"  arg[{i}]: {type(arg).__name__}, len={len(arg)}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}, value={str(arg)[:100]}")

    # --- Phase 2: Determine scenario and execute ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B detected: Factory/Closure pattern with inner data files.")

        try:
            print("Executing recon_slice with outer args to get operator...")
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected recon_slice to return a callable, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Got callable operator: {type(agent_operator)}")

        # Process each inner data file
        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                inner_data = try_load_pickle(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Executing agent_operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                all_passed = False
            else:
                print(f"PASS: Verification succeeded for {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\nScenario A detected: Simple function call.")

        try:
            print("Executing recon_slice with outer args...")
            result = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        if hasattr(expected, 'shape'):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()