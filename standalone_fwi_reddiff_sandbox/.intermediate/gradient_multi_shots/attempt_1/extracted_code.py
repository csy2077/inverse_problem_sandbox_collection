import sys
import os
import dill
import numpy as np
import traceback
import pickle

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_gradient_multi_shots import gradient_multi_shots
from verification_utils import recursive_check


def try_load_data(filepath):
    """Try multiple methods to load the data file."""
    errors = []
    
    # Check file exists and has content
    if not os.path.exists(filepath):
        return None, f"File does not exist: {filepath}"
    
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return None, f"File is empty (0 bytes): {filepath}"
    
    print(f"  File size: {file_size} bytes")
    
    # Method 1: dill with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data, None
    except Exception as e:
        errors.append(f"dill.load: {e}")
    
    # Method 2: pickle with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        errors.append(f"pickle.load: {e}")
    
    # Method 3: dill.loads on bytes
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        data = dill.loads(raw)
        return data, None
    except Exception as e:
        errors.append(f"dill.loads: {e}")
    
    # Method 4: Try loading multiple objects (sometimes file has multiple dumps)
    try:
        objects = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    obj = dill.load(f)
                    objects.append(obj)
                except EOFError:
                    break
        if objects:
            if len(objects) == 1:
                return objects[0], None
            # If multiple objects, try to reconstruct as a dict
            return objects, None
    except Exception as e:
        errors.append(f"multi-object dill.load: {e}")

    # Method 5: numpy load
    try:
        data = np.load(filepath, allow_pickle=True)
        if hasattr(data, 'item'):
            return data.item(), None
        return data, None
    except Exception as e:
        errors.append(f"np.load: {e}")

    return None, f"All load methods failed: {'; '.join(errors)}"


def find_standard_data_file(data_paths):
    """Look for the data file, also check for 'standard_data_' prefix variant."""
    # First check provided paths
    for p in data_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    
    # Check for 'standard_data_' prefix variant in same directory
    for p in data_paths:
        dirn = os.path.dirname(p)
        basen = os.path.basename(p)
        # Try standard_data_ prefix
        std_name = 'standard_' + basen
        std_path = os.path.join(dirn, std_name)
        if os.path.exists(std_path) and os.path.getsize(std_path) > 0:
            return std_path
    
    # List all pkl files in the directory for debugging
    for p in data_paths:
        dirn = os.path.dirname(p)
        if os.path.isdir(dirn):
            all_files = [f for f in os.listdir(dirn) if 'gradient_multi_shots' in f]
            print(f"  Available files matching 'gradient_multi_shots' in {dirn}:")
            for f in sorted(all_files):
                fpath = os.path.join(dirn, f)
                print(f"    {f} ({os.path.getsize(fpath)} bytes)")
            # Try to find any working file
            for f in sorted(all_files):
                fpath = os.path.join(dirn, f)
                if os.path.getsize(fpath) > 0:
                    if 'parent' not in f:
                        return fpath
    
    return None


def find_all_related_files(data_paths):
    """Find all related pkl files in the data directory."""
    outer_paths = []
    inner_paths = []
    
    # Get directory from data_paths
    dirs_checked = set()
    for p in data_paths:
        dirn = os.path.dirname(p)
        if dirn in dirs_checked:
            continue
        dirs_checked.add(dirn)
        if not os.path.isdir(dirn):
            continue
        for f in os.listdir(dirn):
            if 'gradient_multi_shots' not in f:
                continue
            if not f.endswith('.pkl'):
                continue
            fpath = os.path.join(dirn, f)
            if os.path.getsize(fpath) == 0:
                continue
            if 'parent_function' in f or 'parent_' in f:
                inner_paths.append(fpath)
            else:
                outer_paths.append(fpath)
    
    return outer_paths, inner_paths


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_gradient_multi_shots.pkl'
    ]

    # Find all related files
    outer_candidates, inner_paths = find_all_related_files(data_paths)
    
    print(f"Found outer candidates: {[os.path.basename(p) for p in outer_candidates]}")
    print(f"Found inner paths: {[os.path.basename(p) for p in inner_paths]}")

    # Find a working outer data file
    outer_path = None
    outer_data = None
    
    # Prefer the standard_data_ prefixed version
    for p in outer_candidates:
        if 'standard_data_' in os.path.basename(p):
            data, err = try_load_data(p)
            if err is None:
                outer_path = p
                outer_data = data
                break
    
    # Fall back to any loadable outer file
    if outer_data is None:
        for p in outer_candidates:
            data, err = try_load_data(p)
            if err is None:
                outer_path = p
                outer_data = data
                break

    # If still no outer data, try the original path anyway
    if outer_data is None:
        orig_path = data_paths[0]
        print(f"\nAttempting to load original path: {orig_path}")
        data, err = try_load_data(orig_path)
        if err is None:
            outer_path = orig_path
            outer_data = data
        else:
            # Try standard_ prefix
            dirn = os.path.dirname(orig_path)
            std_path = os.path.join(dirn, 'standard_' + os.path.basename(orig_path))
            print(f"Attempting standard_ prefix: {std_path}")
            data, err = try_load_data(std_path)
            if err is None:
                outer_path = std_path
                outer_data = data
            else:
                print(f"FAIL: Could not load any data file.")
                print(f"Last error: {err}")
                # List directory contents for debugging
                if os.path.isdir(dirn):
                    print(f"\nAll .pkl files in {dirn}:")
                    for f in sorted(os.listdir(dirn)):
                        if f.endswith('.pkl'):
                            fpath = os.path.join(dirn, f)
                            print(f"  {f} ({os.path.getsize(fpath)} bytes)")
                sys.exit(1)

    print(f"\nUsing outer data from: {outer_path}")
    
    # Handle case where loaded data might be a list (multiple objects)
    if isinstance(outer_data, list) and len(outer_data) > 0 and not isinstance(outer_data[0], dict):
        # Might be list of objects, check if first has expected keys
        pass
    
    if isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    else:
        print(f"Outer data type: {type(outer_data)}")
        # Try to use it as-is
        if hasattr(outer_data, 'args'):
            outer_args = outer_data.args
            outer_kwargs = getattr(outer_data, 'kwargs', {})
            outer_output = getattr(outer_data, 'output', None)
        else:
            print(f"FAIL: Unexpected data format: {type(outer_data)}")
            sys.exit(1)

    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if inner_paths:
        # --- Scenario B: Factory/Closure pattern ---
        print("\nScenario B detected: Factory/Closure pattern with inner data files.")

        try:
            print("Running gradient_multi_shots to get operator...")
            agent_operator = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Operator obtained, type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            inner_data, err = try_load_data(inner_path)
            if err is not None:
                print(f"FAIL: Could not load inner data file: {err}")
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Executing operator with inner data...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for inner data {os.path.basename(inner_path)}")

    else:
        # --- Scenario A: Simple function call ---
        print("\nScenario A detected: Simple function call.")

        try:
            print("Running gradient_multi_shots...")
            result = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Result obtained, type: {type(result)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

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

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()