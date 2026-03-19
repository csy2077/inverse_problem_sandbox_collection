import sys
import os
import dill
import numpy as np
import traceback
import pickle

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add run_code directory to path
run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)


def try_load_pkl(filepath):
    """Try multiple strategies to load a pkl file."""
    errors = []

    # Check file exists and has content
    if not os.path.exists(filepath):
        return None, [f"File does not exist: {filepath}"]

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return None, [f"File is empty (0 bytes): {filepath}"]

    print(f"  File size: {file_size} bytes")

    # Read raw bytes once
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()

    print(f"  Raw bytes length: {len(raw_bytes)}")
    print(f"  First 20 bytes: {raw_bytes[:20]}")

    # Strategy 1: dill.loads on raw bytes
    try:
        data = dill.loads(raw_bytes)
        print("  Success with dill.loads(raw_bytes)")
        return data, []
    except Exception as e:
        errors.append(f"dill.loads: {e}")

    # Strategy 2: pickle.loads on raw bytes
    try:
        data = pickle.loads(raw_bytes)
        print("  Success with pickle.loads(raw_bytes)")
        return data, []
    except Exception as e:
        errors.append(f"pickle.loads: {e}")

    # Strategy 3: dill.load with file handle
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print("  Success with dill.load(file)")
        return data, []
    except Exception as e:
        errors.append(f"dill.load(file): {e}")

    # Strategy 4: Try loading multiple objects (sometimes files have multiple pickled objects)
    try:
        objects = []
        import io
        buf = io.BytesIO(raw_bytes)
        while True:
            try:
                obj = dill.load(buf)
                objects.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if objects:
            if len(objects) == 1:
                print(f"  Success loading 1 object via sequential dill.load")
                return objects[0], []
            else:
                print(f"  Success loading {len(objects)} objects via sequential dill.load")
                return objects, []
    except Exception as e:
        errors.append(f"sequential dill.load: {e}")

    # Strategy 5: Try with different protocols / recurse settings
    try:
        dill.settings['recurse'] = True
        data = dill.loads(raw_bytes)
        print("  Success with dill.loads recurse=True")
        dill.settings['recurse'] = False
        return data, []
    except Exception as e:
        dill.settings['recurse'] = False
        errors.append(f"dill.loads recurse: {e}")

    # Strategy 6: Check if it's gzipped
    if raw_bytes[:2] == b'\x1f\x8b':
        try:
            import gzip
            decompressed = gzip.decompress(raw_bytes)
            data = dill.loads(decompressed)
            print("  Success with gzip + dill.loads")
            return data, []
        except Exception as e:
            errors.append(f"gzip+dill: {e}")

    # Strategy 7: Check if file might be truncated or has multiple sections
    # Look for pickle protocol headers
    try:
        import struct
        # Check for standard pickle opcodes at various positions
        for offset in range(min(len(raw_bytes), 100)):
            if raw_bytes[offset:offset+1] == b'\x80':  # PROTO opcode
                try:
                    data = dill.loads(raw_bytes[offset:])
                    print(f"  Success with dill.loads at offset {offset}")
                    return data, []
                except:
                    pass
    except Exception as e:
        errors.append(f"offset scan: {e}")

    # Strategy 8: Try numpy load in case it's an npz/npy
    try:
        import io
        data = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        print("  Success with np.load")
        if hasattr(data, 'item'):
            return data.item(), []
        return data, []
    except Exception as e:
        errors.append(f"np.load: {e}")

    return None, errors


def find_all_related_files(std_data_dir, func_name='recon_slice'):
    """Find all pkl files related to the function."""
    related = []
    if os.path.isdir(std_data_dir):
        for fname in sorted(os.listdir(std_data_dir)):
            if fname.endswith('.pkl') and func_name in fname:
                related.append(os.path.join(std_data_dir, fname))
    return related


def main():
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data'

    # Find all related files
    all_files = find_all_related_files(std_data_dir, 'recon_slice')
    print(f"All related pkl files: {all_files}")

    # Also list ALL files in std_data_dir for debugging
    if os.path.isdir(std_data_dir):
        all_dir_files = sorted(os.listdir(std_data_dir))
        print(f"All files in std_data_dir: {all_dir_files}")
        # Show sizes
        for fname in all_dir_files:
            fpath = os.path.join(std_data_dir, fname)
            fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 'DIR'
            print(f"  {fname}: {fsize} bytes")

    # The main data path
    primary_path = os.path.join(std_data_dir, 'data_recon_slice.pkl')

    # Check if there's a "standard_data" variant
    standard_path = os.path.join(std_data_dir, 'standard_data_recon_slice.pkl')
    if os.path.exists(standard_path) and os.path.getsize(standard_path) > 0:
        primary_path = standard_path

    # If primary is empty/missing, try alternatives
    if not os.path.exists(primary_path) or os.path.getsize(primary_path) == 0:
        print(f"Primary path {primary_path} is empty or missing, searching alternatives...")
        for f in all_files:
            if os.path.getsize(f) > 0 and 'parent' not in os.path.basename(f):
                primary_path = f
                break

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in all_files:
        basename = os.path.basename(p)
        if os.path.getsize(p) == 0:
            print(f"Skipping empty file: {basename}")
            continue
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif outer_path is None:
            outer_path = p

    if outer_path is None:
        # Try the primary path even if 0 bytes - to give a clear error
        outer_path = primary_path

    print(f"\nOuter path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # --- Load outer data ---
    print(f"\nLoading outer data from: {outer_path}")
    outer_data, load_errors = try_load_pkl(outer_path)

    if outer_data is None:
        print(f"FAIL: Could not load outer data file.")
        for err in load_errors:
            print(f"  Error: {err}")

        # If the file is 0 bytes, it may have been a write failure during data generation.
        # Try to regenerate the data by running the gen code if available.
        file_size = os.path.getsize(outer_path) if os.path.exists(outer_path) else -1
        if file_size == 0:
            print(f"\nFile is 0 bytes. Attempting to find and run generation code...")

            # Look for gen scripts
            gen_candidates = []
            for d in [run_code_dir, os.path.dirname(std_data_dir)]:
                if os.path.isdir(d):
                    for fname in os.listdir(d):
                        if 'gen' in fname.lower() and fname.endswith('.py'):
                            gen_candidates.append(os.path.join(d, fname))

            print(f"Gen script candidates: {gen_candidates}")

            # Try running gen_data scripts
            for gen_script in gen_candidates:
                print(f"Attempting to run: {gen_script}")
                try:
                    exec(open(gen_script).read(), {'__name__': '__gen__'})
                    # Check if file now has content
                    if os.path.exists(outer_path) and os.path.getsize(outer_path) > 0:
                        print("Gen script produced data, retrying load...")
                        outer_data, load_errors = try_load_pkl(outer_path)
                        if outer_data is not None:
                            break
                except Exception as e:
                    print(f"  Gen script failed: {e}")

        if outer_data is None:
            # Last resort: try to find ANY loadable pkl in the directory
            print("\nLast resort: trying all pkl files in the directory...")
            for fname in sorted(os.listdir(std_data_dir)):
                if not fname.endswith('.pkl'):
                    continue
                fpath = os.path.join(std_data_dir, fname)
                if os.path.getsize(fpath) == 0:
                    continue
                print(f"  Trying: {fname}")
                test_data, _ = try_load_pkl(fpath)
                if test_data is not None and isinstance(test_data, dict):
                    fn = test_data.get('func_name', '')
                    if 'recon_slice' in str(fn):
                        print(f"  Found matching data in {fname}!")
                        outer_data = test_data
                        outer_path = fpath
                        break

        if outer_data is None:
            print("FAIL: All loading attempts exhausted.")
            sys.exit(1)

    print(f"Outer data type: {type(outer_data)}")
    if isinstance(outer_data, dict):
        print(f"Outer data keys: {list(outer_data.keys())}")
    elif isinstance(outer_data, (list, tuple)):
        print(f"Outer data length: {len(outer_data)}")
        # Maybe it's a list with one dict
        if len(outer_data) > 0 and isinstance(outer_data[0], dict):
            outer_data = outer_data[0]
            print(f"Unwrapped to first element, keys: {list(outer_data.keys())}")

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
                if hasattr(arg, '__class__'):
                    print(f"    class: {arg.__class__.__name__}")
                    if hasattr(arg, '__dict__'):
                        print(f"    attrs: {list(arg.__dict__.keys())[:10]}")

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
                inner_data, inner_errors = try_load_pkl(inner_path)
                if inner_data is None:
                    print(f"FAIL: Could not load inner data: {inner_errors}")
                    sys.exit(1)
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