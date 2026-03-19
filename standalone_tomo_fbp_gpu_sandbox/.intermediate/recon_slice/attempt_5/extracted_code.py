import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add run_code directory to path
run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Scan directory for related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        all_related = []
        for fname in sorted(os.listdir(std_data_dir)):
            if fname.endswith('.pkl') and 'recon_slice' in fname:
                full_path = os.path.join(std_data_dir, fname)
                if full_path not in data_paths:
                    data_paths.append(full_path)
                all_related.append(full_path)
        print(f"All related pkl files found: {all_related}")

    print(f"All data paths: {data_paths}")

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename == 'data_recon_slice.pkl' or basename == 'standard_data_recon_slice.pkl':
            outer_path = p
        elif 'recon_slice' in basename and 'parent' not in basename:
            if outer_path is None:
                outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    print(f"\nOuter path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # --- Load outer data ---
    # The error "function() argument 2 must be dict, not module" comes from
    # dill trying to reconstruct a function and getting confused when
    # builtins.__main__ is set to a module object.
    # The fix: do NOT set builtins.__main__. Instead, use plain dill.load
    # with proper file handling.

    print(f"\nLoading outer data from: {outer_path}")

    outer_data = None
    load_error = None

    # The previous error "Ran out of input" with dill.loads was because
    # we read the file and then tried loads on it, but dill.load from file handle
    # might work differently. The "function() argument 2 must be dict, not module"
    # error with pickle is because pickle can't handle dill-serialized data properly.
    # We should ONLY use dill.load with a file handle.

    # Attempt 1: Standard dill.load with file handle
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("Success with dill.load(file)")
    except Exception as e:
        load_error = str(e)
        print(f"  dill.load(file) failed: {e}")
        traceback.print_exc()

    # Attempt 2: dill.loads with full bytes
    if outer_data is None:
        try:
            with open(outer_path, 'rb') as f:
                raw = f.read()
            outer_data = dill.loads(raw)
            print("Success with dill.loads(bytes)")
        except Exception as e:
            print(f"  dill.loads(bytes) failed: {e}")

    # Attempt 3: Try with dill settings
    if outer_data is None:
        try:
            dill.settings['recurse'] = True
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print("Success with dill.load(file) recurse=True")
        except Exception as e:
            print(f"  dill.load recurse failed: {e}")
        finally:
            dill.settings['recurse'] = False

    if outer_data is None:
        print(f"FAIL: Could not load outer data file.")
        print(f"Last error: {load_error}")
        sys.exit(1)

    print(f"Outer data type: {type(outer_data)}")
    print(f"Outer data keys: {list(outer_data.keys()) if isinstance(outer_data, dict) else 'N/A'}")

    # Extract args/kwargs/output
    if not isinstance(outer_data, dict):
        print(f"FAIL: outer_data is not a dict, it's {type(outer_data)}")
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args) if outer_args else 0}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else []}")

    # Print arg info for debugging
    if outer_args:
        for i, arg in enumerate(outer_args):
            if hasattr(arg, 'shape'):
                print(f"  arg[{i}]: {type(arg).__name__}, shape={arg.shape}, dtype={getattr(arg, 'dtype', 'N/A')}")
            elif isinstance(arg, str):
                print(f"  arg[{i}]: str = '{arg}'")
            else:
                print(f"  arg[{i}]: {type(arg).__name__}")

    # Import function and verification
    from agent_recon_slice import recon_slice
    from verification_utils import recursive_check

    # Determine scenario
    if inner_paths:
        # --- Scenario B: Factory/Closure ---
        print("\nDetected Scenario B (Factory/Closure pattern)")
        try:
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice (outer call) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from recon_slice, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not isinstance(inner_data, dict):
                print(f"FAIL: inner_data is not a dict, it's {type(inner_data)}")
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception: {e}")
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
                sys.exit(1)
            else:
                print(f"PASS: {os.path.basename(inner_path)} verified.")
    else:
        # --- Scenario A: Simple function call ---
        print("\nDetected Scenario A (Simple function call)")
        try:
            result = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice raised an exception: {e}")
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
            print("PASS: Output verified successfully.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()