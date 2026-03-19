import sys
import os
import dill
import numpy as np
import traceback
import pickle

# Import the target function
from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def load_data(filepath):
    """Try multiple approaches to load a pickle/dill file."""
    # Try dill first
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception:
        pass

    # Try pickle
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception:
        pass

    # Try dill with different protocols
    try:
        with open(filepath, 'rb') as f:
            unpickler = dill.Unpickler(f)
            data = unpickler.load()
        return data
    except Exception:
        pass

    # Try reading all bytes first, then loading from bytes
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        data = dill.loads(raw)
        return data
    except Exception:
        pass

    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        data = pickle.loads(raw)
        return data
    except Exception:
        pass

    # Try loading multiple objects (sometimes files have multiple dumps)
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
    except Exception:
        pass

    raise RuntimeError(f"Could not load data from {filepath} with any method")


def find_data_files(base_dir, func_name):
    """Search for data files in the directory."""
    candidates = []
    if not os.path.isdir(base_dir):
        return candidates
    for fname in os.listdir(base_dir):
        if fname.endswith('.pkl') and func_name in fname:
            candidates.append(os.path.join(base_dir, fname))
    return candidates


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_nnfbp_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Check file sizes and validity
    valid_paths = []
    for p in data_paths:
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"File: {p}, size: {size} bytes")
            if size > 0:
                valid_paths.append(p)
            else:
                print(f"WARNING: File {p} is empty (0 bytes)")
        else:
            print(f"WARNING: File {p} does not exist")

    # Also search the std_data directory for any related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_nnfbp_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        print(f"\nFiles in std_data directory:")
        all_files = []
        for fname in sorted(os.listdir(std_data_dir)):
            fpath = os.path.join(std_data_dir, fname)
            fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            print(f"  {fname} ({fsize} bytes)")
            if fname.endswith('.pkl') and 'recon_slice' in fname and fsize > 0:
                if fpath not in valid_paths:
                    valid_paths.append(fpath)
                all_files.append(fpath)

    if not valid_paths:
        # Try alternate naming patterns
        alt_patterns = [
            os.path.join(std_data_dir, 'standard_data_recon_slice.pkl'),
            os.path.join(std_data_dir, 'data_recon_slice.pkl'),
        ]
        for alt in alt_patterns:
            if os.path.exists(alt) and os.path.getsize(alt) > 0:
                valid_paths.append(alt)

    if not valid_paths:
        print("FAIL: No valid data files found")
        sys.exit(1)

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in valid_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif 'recon_slice' in basename:
            if outer_path is None:
                outer_path = p
            else:
                # Pick the one without "parent" and with "standard" or just "data_"
                if 'standard' in basename:
                    outer_path = p

    if outer_path is None and valid_paths:
        # Use the first valid path as outer
        outer_path = valid_paths[0]

    if outer_path is None:
        print("FAIL: Could not find any valid outer data file")
        sys.exit(1)

    print(f"\nUsing outer_path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # Phase 1: Load outer data
    try:
        outer_data = load_data(outer_path)
        print(f"Loaded outer data successfully")
        if isinstance(outer_data, dict):
            print(f"  Keys: {list(outer_data.keys())}")
            print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        elif isinstance(outer_data, list):
            print(f"  Loaded a list of {len(outer_data)} items")
            # If it's a list, the first element might be the dict
            if len(outer_data) > 0 and isinstance(outer_data[0], dict):
                outer_data = outer_data[0]
                print(f"  Using first element, keys: {list(outer_data.keys())}")
        else:
            print(f"  Type: {type(outer_data)}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(outer_data, dict):
        print(f"FAIL: Expected dict from data file, got {type(outer_data)}")
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # Print some debug info about args
    print(f"\nOuter args count: {len(outer_args)}")
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nDetected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
            print("Phase 1: recon_slice returned an operator successfully.")
        except Exception as e:
            print(f"FAIL: recon_slice raised an exception during Phase 1: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator but got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute the operator
        all_passed = True
        for inner_path in inner_paths:
            try:
                inner_data = load_data(inner_path)
                print(f"Loaded inner data from: {inner_path}")
                if isinstance(inner_data, list) and len(inner_data) > 0 and isinstance(inner_data[0], dict):
                    inner_data = inner_data[0]
            except Exception as e:
                print(f"FAIL: Could not load inner data file {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}")
                print(f"  Message: {msg}")
                all_passed = False
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\nDetected Scenario A: Simple function call")

        try:
            actual_result = recon_slice(*outer_args, **outer_kwargs)
            print("recon_slice executed successfully.")
            if isinstance(actual_result, np.ndarray):
                print(f"  Result: ndarray shape={actual_result.shape}, dtype={actual_result.dtype}")
            else:
                print(f"  Result type: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: recon_slice raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if expected_output is not None:
            if isinstance(expected_output, np.ndarray):
                print(f"  Expected: ndarray shape={expected_output.shape}, dtype={expected_output.dtype}")
            else:
                print(f"  Expected type: {type(expected_output)}")

        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()