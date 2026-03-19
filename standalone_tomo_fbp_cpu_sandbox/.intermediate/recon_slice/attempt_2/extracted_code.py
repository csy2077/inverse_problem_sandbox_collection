import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


class Python2Unpickler(pickle.Unpickler):
    """Custom unpickler that handles Python 2 pickles in Python 3."""
    
    def find_class(self, module, name):
        # Map Python 2 module names to Python 3 equivalents
        module_mapping = {
            '__builtin__': 'builtins',
            'copy_reg': 'copyreg',
            'Queue': 'queue',
            'repr': 'reprlib',
            'UserDict': 'collections',
            'UserList': 'collections',
            'UserString': 'collections',
            'itertools': 'itertools',
        }
        
        if module in module_mapping:
            module = module_mapping[module]
        
        return super().find_class(module, name)


class DillPython2Unpickler(dill.Unpickler):
    """Custom dill unpickler that handles Python 2 pickles."""
    
    def find_class(self, module, name):
        module_mapping = {
            '__builtin__': 'builtins',
            'copy_reg': 'copyreg',
            'Queue': 'queue',
            'repr': 'reprlib',
            'UserDict': 'collections',
            'UserList': 'collections',
            'UserString': 'collections',
        }
        
        if module in module_mapping:
            module = module_mapping[module]
        
        return super().find_class(module, name)


def load_data(filepath):
    """Try multiple methods to load a pickle/dill file, handling Python 2 pickles."""
    errors = []
    file_size = os.path.getsize(filepath)
    
    # Read raw bytes first to check for truncation / multiple objects
    with open(filepath, 'rb') as f:
        raw_data = f.read()
    
    # Check the pickle protocol from header
    # \x80\x03 means protocol 3, which is Python 3 compatible
    # But the error says "__builtin__" which is Python 2
    # The file header shows protocol 3 (\x80\x03) but contains Python 2 references
    
    # Method 1: Custom Python2 pickle unpickler on raw bytes
    try:
        unpickler = Python2Unpickler(io.BytesIO(raw_data))
        data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"Python2Unpickler: {e}")

    # Method 2: Custom Dill Python2 unpickler on raw bytes
    try:
        unpickler = DillPython2Unpickler(io.BytesIO(raw_data))
        data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"DillPython2Unpickler: {e}")

    # Method 3: pickle with encoding='latin1' (standard py2->py3 approach)
    try:
        data = pickle.loads(raw_data, encoding='latin1')
        return data
    except Exception as e:
        errors.append(f"pickle.loads latin1: {e}")

    # Method 4: pickle with encoding='bytes'
    try:
        data = pickle.loads(raw_data, encoding='bytes')
        return data
    except Exception as e:
        errors.append(f"pickle.loads bytes: {e}")

    # Method 5: dill.loads on raw bytes
    try:
        data = dill.loads(raw_data)
        return data
    except Exception as e:
        errors.append(f"dill.loads: {e}")

    # Method 6: Try with file handle using dill
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"dill.load file: {e}")

    # Method 7: Standard pickle load
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        errors.append(f"pickle.load file: {e}")

    # Method 8: Try to find and fix the truncation issue
    # The "Ran out of input" from dill suggests the file might have 
    # been saved with pickle but loaded with dill, or vice versa
    # Let's try reading just enough
    try:
        buf = io.BytesIO(raw_data)
        unpickler = pickle.Unpickler(buf, encoding='latin1')
        data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"pickle.Unpickler latin1: {e}")

    # Method 9: Try scanning for multiple concatenated pickles
    try:
        buf = io.BytesIO(raw_data)
        results = []
        while buf.tell() < len(raw_data):
            try:
                unpickler = Python2Unpickler(buf)
                obj = unpickler.load()
                results.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if results:
            return results[0] if len(results) == 1 else results
    except Exception as e:
        errors.append(f"multi-pickle scan: {e}")

    # Method 10: numpy load
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        errors.append(f"np.load: {e}")

    raise RuntimeError(
        f"All load methods failed for {filepath} (size={file_size} bytes). Errors: {errors}"
    )


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Also search the directory for any related files we might have missed
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        for f in sorted(os.listdir(std_data_dir)):
            full_path = os.path.join(std_data_dir, f)
            if full_path not in data_paths and 'recon_slice' in f:
                data_paths.append(full_path)
                print(f"Discovered additional data file: {f}")

    # Print file info for debugging
    for p in data_paths:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"File: {os.path.basename(p)}, size: {sz} bytes")
        else:
            print(f"File NOT FOUND: {p}")

    # Classify paths into outer (standard) and inner (parent_function) categories
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if not os.path.exists(p):
            continue
        if os.path.getsize(p) == 0:
            print(f"WARNING: Skipping empty file: {p}")
            continue
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No valid outer data file found.")
        sys.exit(1)

    # Check file size
    file_size = os.path.getsize(outer_path)
    print(f"Outer data file size: {file_size} bytes")

    if file_size == 0:
        print("FAIL: Outer data file is empty (0 bytes).")
        sys.exit(1)

    # --- Phase 1: Load outer data and execute recon_slice ---
    print(f"Loading outer data from: {outer_path}")
    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()

        # Debug: print first bytes of the file
        try:
            with open(outer_path, 'rb') as f:
                header = f.read(min(200, file_size))
            print(f"File header (hex): {header[:100].hex()}")
            print(f"File header (raw): {header[:100]}")
        except:
            pass
        sys.exit(1)

    # Handle case where loaded data might be a list
    if isinstance(outer_data, list):
        if len(outer_data) == 1:
            outer_data = outer_data[0]
        elif len(outer_data) > 1:
            for item in outer_data:
                if isinstance(item, dict) and 'args' in item:
                    outer_data = item
                    break

    # Handle bytes keys from Python 2 pickle loading
    if isinstance(outer_data, dict):
        # Check if keys are bytes instead of strings
        sample_keys = list(outer_data.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_data = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_data.items()
            }

    if not isinstance(outer_data, dict):
        print(f"WARNING: outer_data is not a dict, it's {type(outer_data)}")
        if hasattr(outer_data, 'item'):
            outer_data = outer_data.item()

    outer_args = outer_data.get('args', ()) if isinstance(outer_data, dict) else ()
    outer_kwargs = outer_data.get('kwargs', {}) if isinstance(outer_data, dict) else {}
    outer_output = outer_data.get('output', None) if isinstance(outer_data, dict) else None

    # Handle bytes keys in kwargs
    if isinstance(outer_kwargs, dict):
        sample_keys = list(outer_kwargs.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_kwargs = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_kwargs.items()
            }

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A') if isinstance(outer_data, dict) else 'N/A'}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Debug: print types of args
    for i, arg in enumerate(outer_args):
        print(f"  arg[{i}] type: {type(arg)}, ", end="")
        if isinstance(arg, np.ndarray):
            print(f"shape: {arg.shape}, dtype: {arg.dtype}")
        elif isinstance(arg, str):
            print(f"value: {arg}")
        else:
            print(f"class: {arg.__class__.__name__}")

    try:
        agent_result = recon_slice(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: recon_slice execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Determine scenario and verify ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected recon_slice to return a callable (operator), got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                inner_data = load_data(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            if isinstance(inner_data, list) and len(inner_data) == 1:
                inner_data = inner_data[0]

            # Handle bytes keys
            if isinstance(inner_data, dict):
                sample_keys = list(inner_data.keys())
                if sample_keys and isinstance(sample_keys[0], bytes):
                    inner_data = {
                        (k.decode('utf-8') if isinstance(k, bytes) else k): v
                        for k, v in inner_data.items()
                    }

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if isinstance(inner_kwargs, dict):
                sample_keys = list(inner_kwargs.keys())
                if sample_keys and isinstance(sample_keys[0], bytes):
                    inner_kwargs = {
                        (k.decode('utf-8') if isinstance(k, bytes) else k): v
                        for k, v in inner_kwargs.items()
                    }

            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL (inner test {idx + 1}): {msg}")
                all_passed = False
            else:
                print(f"PASS (inner test {idx + 1})")

        if not all_passed:
            sys.exit(1)
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function, comparing output directly.")

        expected = outer_output
        result = agent_result

        print(f"Result type: {type(result)}")
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")

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