import sys
import os
import dill
import numpy as np
import traceback
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_forward_multi_shots import forward_multi_shots
from verification_utils import recursive_check


def try_load_data(filepath):
    """Try multiple methods to load the data file."""
    errors = []
    
    # Check file exists and has content
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_size = os.path.getsize(filepath)
    print(f"  File size: {file_size} bytes")
    
    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    # Method 1: dill with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print("  Loaded successfully with dill (rb)")
        return data
    except Exception as e:
        errors.append(f"dill(rb): {e}")
    
    # Method 2: pickle with 'rb'
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print("  Loaded successfully with pickle (rb)")
        return data
    except Exception as e:
        errors.append(f"pickle(rb): {e}")
    
    # Method 3: dill.loads on bytes
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        data = dill.loads(raw)
        print("  Loaded successfully with dill.loads")
        return data
    except Exception as e:
        errors.append(f"dill.loads: {e}")
    
    # Method 4: Try loading multiple objects (file may have been appended to)
    try:
        results = []
        with open(filepath, 'rb') as f:
            while True:
                try:
                    obj = dill.load(f)
                    results.append(obj)
                except EOFError:
                    break
        if results:
            # Return the last valid object, or if there's just one, return it
            print(f"  Loaded {len(results)} object(s) via sequential dill.load")
            return results[-1] if len(results) == 1 else results[-1]
    except Exception as e:
        errors.append(f"sequential dill: {e}")

    # Method 5: Try with different pickle protocols
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        # Find pickle STOP opcode (0x2e = '.') boundaries
        # Try loading from the beginning with different unpicklers
        import io
        buf = io.BytesIO(raw)
        data = dill.Unpickler(buf).load()
        print("  Loaded successfully with dill.Unpickler")
        return data
    except Exception as e:
        errors.append(f"dill.Unpickler: {e}")

    # Method 6: numpy load
    try:
        data = np.load(filepath, allow_pickle=True)
        print("  Loaded successfully with numpy")
        return data.item() if data.ndim == 0 else data
    except Exception as e:
        errors.append(f"numpy: {e}")

    raise RuntimeError(f"All loading methods failed for {filepath}:\n" + "\n".join(errors))


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_forward_multi_shots.pkl'
    ]

    # Also scan the directory for any related files we might have missed
    std_data_dir = os.path.dirname(data_paths[0])
    all_related_files = []
    if os.path.isdir(std_data_dir):
        for fname in os.listdir(std_data_dir):
            if 'forward_multi_shots' in fname and fname.endswith('.pkl'):
                full_path = os.path.join(std_data_dir, fname)
                if full_path not in data_paths:
                    all_related_files.append(full_path)
                    print(f"  Discovered additional related file: {fname}")
        data_paths.extend(all_related_files)

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'data_forward_multi_shots.pkl' or basename == 'standard_data_forward_multi_shots.pkl':
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file")
        sys.exit(1)

    # Check file details
    print(f"Outer data file: {outer_path}")
    print(f"  Exists: {os.path.exists(outer_path)}")
    if os.path.exists(outer_path):
        print(f"  Size: {os.path.getsize(outer_path)} bytes")

    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        outer_data = try_load_data(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Handle case where outer_data might be in different formats
    if isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
    elif isinstance(outer_data, (list, tuple)):
        # Maybe it's a list of dicts or a tuple of (args, kwargs, output)
        if len(outer_data) >= 3:
            outer_args = outer_data[0] if isinstance(outer_data[0], (list, tuple)) else (outer_data[0],)
            outer_kwargs = outer_data[1] if isinstance(outer_data[1], dict) else {}
            outer_output = outer_data[2]
        elif len(outer_data) == 1 and isinstance(outer_data[0], dict):
            outer_args = outer_data[0].get('args', ())
            outer_kwargs = outer_data[0].get('kwargs', {})
            outer_output = outer_data[0].get('output', None)
        else:
            print(f"FAIL: Unexpected outer data format: {type(outer_data)}, length: {len(outer_data)}")
            sys.exit(1)
    else:
        print(f"FAIL: Unexpected outer data type: {type(outer_data)}")
        print(f"  Content preview: {str(outer_data)[:500]}")
        sys.exit(1)

    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A') if isinstance(outer_data, dict) else 'N/A'}")
    print(f"  args count: {len(outer_args)}")
    print(f"  kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        print("Phase 1: Reconstructing operator via forward_multi_shots(...)...")
        try:
            agent_operator = forward_multi_shots(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_multi_shots raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Operator created successfully: {type(agent_operator)}")

        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                inner_data = try_load_data(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            print(f"  inner args count: {len(inner_args)}")
            print(f"  inner kwargs keys: {list(inner_kwargs.keys())}")

            print("Executing operator with inner args...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Result mismatch for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        print("Executing forward_multi_shots(...)...")
        try:
            result = forward_multi_shots(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        print("Comparing results...")
        print(f"  Expected type: {type(expected)}")
        print(f"  Result type: {type(result)}")

        # Debug: print more info about the results
        if isinstance(expected, list):
            print(f"  Expected list length: {len(expected)}")
            if len(expected) > 0:
                print(f"  Expected[0] type: {type(expected[0])}")
        if isinstance(result, list):
            print(f"  Result list length: {len(result)}")
            if len(result) > 0:
                print(f"  Result[0] type: {type(result[0])}")

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Result mismatch")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("PASS: Output verified successfully.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()