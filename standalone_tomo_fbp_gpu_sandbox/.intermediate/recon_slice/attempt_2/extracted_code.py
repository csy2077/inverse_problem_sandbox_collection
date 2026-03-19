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


def try_load_data(filepath):
    """Try multiple approaches to load the data file."""
    errors = []

    if not os.path.exists(filepath):
        return None, f"File does not exist: {filepath}"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return None, f"File is empty (0 bytes): {filepath}"

    print(f"File size: {file_size} bytes")

    # Read raw bytes once
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Attempt 1: dill with 'rb'
    try:
        data = dill.loads(raw)
        return data, None
    except Exception as e:
        errors.append(f"dill.loads failed: {e}")

    # Attempt 2: pickle with 'rb'
    try:
        data = pickle.loads(raw)
        return data, None
    except Exception as e:
        errors.append(f"pickle.loads failed: {e}")

    # Attempt 3: Fix __builtin__ -> builtins mapping (Python 2 -> 3 compatibility)
    class Python2Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__builtin__':
                module = 'builtins'
            return super().find_class(module, name)

    try:
        buf = io.BytesIO(raw)
        data = Python2Unpickler(buf).load()
        return data, None
    except Exception as e:
        errors.append(f"Python2Unpickler failed: {e}")

    # Attempt 4: dill Unpickler with __builtin__ fix
    class DillPython2Unpickler(dill.Unpickler):
        def find_class(self, module, name):
            if module == '__builtin__':
                module = 'builtins'
            return super().find_class(module, name)

    try:
        buf = io.BytesIO(raw)
        data = DillPython2Unpickler(buf).load()
        return data, None
    except Exception as e:
        errors.append(f"DillPython2Unpickler failed: {e}")

    # Attempt 5: Try multiple pickle objects concatenated
    try:
        results = []
        buf = io.BytesIO(raw)
        while buf.tell() < len(raw):
            try:
                obj = Python2Unpickler(buf).load()
                results.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if results:
            if len(results) == 1:
                return results[0], None
            # Try to merge dicts
            merged = {}
            for r in results:
                if isinstance(r, dict):
                    merged.update(r)
            if merged:
                return merged, None
            return results, None
    except Exception as e:
        errors.append(f"multi-object Python2Unpickler failed: {e}")

    # Attempt 6: Try multiple dill objects concatenated
    try:
        results = []
        buf = io.BytesIO(raw)
        while buf.tell() < len(raw):
            try:
                obj = DillPython2Unpickler(buf).load()
                results.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if results:
            if len(results) == 1:
                return results[0], None
            merged = {}
            for r in results:
                if isinstance(r, dict):
                    merged.update(r)
            if merged:
                return merged, None
            return results, None
    except Exception as e:
        errors.append(f"multi-object DillPython2Unpickler failed: {e}")

    # Attempt 7: Scan for pickle protocol headers at various offsets
    try:
        for offset in range(min(1024, len(raw))):
            if raw[offset:offset + 1] == b'\x80':
                try:
                    buf = io.BytesIO(raw[offset:])
                    data = Python2Unpickler(buf).load()
                    return data, None
                except Exception:
                    pass
                try:
                    buf = io.BytesIO(raw[offset:])
                    data = DillPython2Unpickler(buf).load()
                    return data, None
                except Exception:
                    pass
    except Exception as e:
        errors.append(f"offset scan failed: {e}")

    # Attempt 8: numpy load
    try:
        data = np.load(filepath, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        return data, None
    except Exception as e:
        errors.append(f"np.load failed: {e}")

    # Attempt 9: Try loading with encoding='latin1' (Python 2 pickles)
    try:
        buf = io.BytesIO(raw)
        data = pickle.load(buf, encoding='latin1')
        return data, None
    except Exception as e:
        errors.append(f"pickle.load encoding=latin1 failed: {e}")

    # Attempt 10: Try with encoding='bytes'
    try:
        buf = io.BytesIO(raw)
        data = pickle.load(buf, encoding='bytes')
        return data, None
    except Exception as e:
        errors.append(f"pickle.load encoding=bytes failed: {e}")

    # Attempt 11: Custom unpickler with encoding='latin1' and __builtin__ fix
    class Python2UnpicklerLatin1(pickle.Unpickler):
        def find_class(self, module, name):
            if module == '__builtin__':
                module = 'builtins'
            # Handle bytes module names
            if isinstance(module, bytes):
                module = module.decode('utf-8')
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            return super().find_class(module, name)

    try:
        buf = io.BytesIO(raw)
        unpickler = Python2UnpicklerLatin1(buf)
        unpickler.encoding = 'latin1'
        data = unpickler.load()
        return data, None
    except Exception as e:
        errors.append(f"Python2UnpicklerLatin1 failed: {e}")

    # Attempt 12: Check if file has multiple concatenated pickles with different protocols
    try:
        results = []
        pos = 0
        while pos < len(raw):
            for attempt_offset in range(min(16, len(raw) - pos)):
                try:
                    buf = io.BytesIO(raw[pos + attempt_offset:])
                    unpickler = Python2Unpickler(buf)
                    obj = unpickler.load()
                    results.append(obj)
                    pos = pos + attempt_offset + buf.tell()
                    break
                except Exception:
                    continue
            else:
                break
        if results:
            if len(results) == 1:
                return results[0], None
            merged = {}
            for r in results:
                if isinstance(r, dict):
                    merged.update(r)
            if merged:
                return merged, None
            return results, None
    except Exception as e:
        errors.append(f"multi-protocol scan failed: {e}")

    return None, "All load attempts failed:\n" + "\n".join(errors)


def inspect_file_header(filepath):
    """Print the first few bytes for debugging."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(64)
        print(f"File header (hex): {header[:32].hex()}")
        print(f"File header (repr): {repr(header[:32])}")
        # Check if it's a zip/npz
        if header[:2] == b'PK':
            print("Detected: ZIP/NPZ format")
        elif header[:1] == b'\x80':
            proto = header[1]
            print(f"Detected: Pickle protocol {proto}")
        elif header[:1] == b'(':
            print("Detected: Pickle protocol 0")
        elif header[:6] == b'\x93NUMPY':
            print("Detected: NumPy .npy format")
        else:
            print(f"Unknown format, first byte: {header[0]}")
    except Exception as e:
        print(f"Could not inspect file: {e}")


def find_all_data_files(base_dir, func_name):
    """Search for all related pkl files in the directory."""
    found = []
    if not os.path.isdir(base_dir):
        return found
    for fname in os.listdir(base_dir):
        if fname.endswith('.pkl') and func_name in fname:
            found.append(os.path.join(base_dir, fname))
    return sorted(found)


def robust_load(filepath):
    """Most robust loading approach - try everything."""
    errors = []

    with open(filepath, 'rb') as f:
        raw = f.read()

    # First, check if this is actually multiple pickled objects
    # The error "Ran out of input" with dill but "__builtin__" with pickle
    # suggests dill might be reading past end or the file has protocol issues

    # Try: dill with different protocol settings
    for proto_fix in [True, False]:
        for enc in ['ASCII', 'latin1', 'bytes', 'utf-8']:
            try:
                buf = io.BytesIO(raw)

                class CustomUnpickler(dill.Unpickler):
                    def find_class(self, module, name):
                        if module == '__builtin__':
                            module = 'builtins'
                        if isinstance(module, bytes):
                            module = module.decode('utf-8', errors='replace')
                        if isinstance(name, bytes):
                            name = name.decode('utf-8', errors='replace')
                        return super().find_class(module, name)

                unpickler = CustomUnpickler(buf)
                try:
                    unpickler.encoding = enc
                except:
                    pass
                data = unpickler.load()
                print(f"Success with CustomUnpickler encoding={enc}")
                return data, None
            except Exception as e:
                errors.append(f"CustomUnpickler enc={enc}: {e}")

    # Try standard pickle with encodings
    for enc in ['latin1', 'bytes', 'ASCII']:
        try:

            class StdUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == '__builtin__':
                        module = 'builtins'
                    return super().find_class(module, name)

            buf = io.BytesIO(raw)
            unpickler = StdUnpickler(buf)
            try:
                unpickler.encoding = enc
            except:
                pass
            data = unpickler.load()
            print(f"Success with StdUnpickler encoding={enc}")
            return data, None
        except Exception as e:
            errors.append(f"StdUnpickler enc={enc}: {e}")

    return None, "\n".join(errors)


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Also scan the directory for any related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        all_related = find_all_data_files(std_data_dir, 'recon_slice')
        print(f"All related pkl files found in std_data dir: {all_related}")
        for p in all_related:
            if p not in data_paths:
                data_paths.append(p)

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

    # Inspect file before loading
    print(f"\nInspecting file: {outer_path}")
    inspect_file_header(outer_path)

    # --- Load outer data ---
    print(f"\nLoading outer data from: {outer_path}")
    
    # First try the standard approach
    outer_data, load_err = try_load_data(outer_path)

    if outer_data is None:
        print(f"Standard load failed, trying robust_load...")
        outer_data, load_err2 = robust_load(outer_path)

    if outer_data is None:
        # Last resort: try to add the std_data directory to path and reload
        sys.path.insert(0, std_data_dir)
        
        # Also try setting up __builtin__ as an alias
        try:
            import builtins
            sys.modules['__builtin__'] = builtins
        except:
            pass

        outer_data, load_err3 = try_load_data(outer_path)
        if outer_data is None:
            outer_data, load_err4 = robust_load(outer_path)

    if outer_data is None:
        # Try alternative path names
        alt_paths = [
            os.path.join(std_data_dir, 'standard_data_recon_slice.pkl'),
            os.path.join(std_data_dir, 'std_data_recon_slice.pkl'),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path) and alt_path != outer_path:
                print(f"Trying alternative path: {alt_path}")
                outer_data, _ = try_load_data(alt_path)
                if outer_data is not None:
                    print("Successfully loaded from alternative path.")
                    break

    if outer_data is None:
        print(f"FAIL: Could not load outer data file after all attempts.")
        print(f"Last error: {load_err}")
        sys.exit(1)

    # Handle list wrapper
    if isinstance(outer_data, list) and not isinstance(outer_data, dict):
        if len(outer_data) == 1:
            outer_data = outer_data[0]
        else:
            for item in outer_data:
                if isinstance(item, dict) and 'args' in item:
                    outer_data = item
                    break

    if not isinstance(outer_data, dict):
        print(f"WARN: outer_data is not a dict, it's {type(outer_data)}.")
        if isinstance(outer_data, (tuple, list)) and len(outer_data) >= 2:
            outer_data = {
                'args': outer_data[0],
                'kwargs': outer_data[1] if len(outer_data) > 1 else {},
                'output': outer_data[2] if len(outer_data) > 2 else None
            }
        else:
            print(f"FAIL: Cannot interpret outer_data of type {type(outer_data)}")
            sys.exit(1)

    # Handle bytes keys (from Python 2 pickles loaded with encoding='bytes')
    if outer_data and isinstance(outer_data, dict):
        new_data = {}
        for k, v in outer_data.items():
            if isinstance(k, bytes):
                new_data[k.decode('utf-8')] = v
            else:
                new_data[k] = v
        outer_data = new_data

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Fix bytes keys in kwargs
    if isinstance(outer_kwargs, dict):
        new_kwargs = {}
        for k, v in outer_kwargs.items():
            if isinstance(k, bytes):
                new_kwargs[k.decode('utf-8')] = v
            else:
                new_kwargs[k] = v
        outer_kwargs = new_kwargs

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args) if outer_args else 0}, kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else []}")

    # Print arg info
    if outer_args:
        for i, arg in enumerate(outer_args):
            if hasattr(arg, 'shape'):
                print(f"  arg[{i}]: {type(arg).__name__}, shape={arg.shape}, dtype={getattr(arg, 'dtype', 'N/A')}")
            elif isinstance(arg, str):
                print(f"  arg[{i}]: str = '{arg}'")
            else:
                print(f"  arg[{i}]: {type(arg).__name__}")

    # Determine scenario
    if inner_paths:
        # --- Scenario B ---
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
            inner_data, inner_err = try_load_data(inner_path)
            if inner_data is None:
                # Try __builtin__ fix
                try:
                    import builtins
                    sys.modules['__builtin__'] = builtins
                except:
                    pass
                inner_data, inner_err = robust_load(inner_path)

            if inner_data is None:
                print(f"FAIL: Could not load inner data file: {inner_err}")
                sys.exit(1)

            if isinstance(inner_data, list) and len(inner_data) == 1:
                inner_data = inner_data[0]

            # Fix bytes keys
            if isinstance(inner_data, dict):
                new_data = {}
                for k, v in inner_data.items():
                    if isinstance(k, bytes):
                        new_data[k.decode('utf-8')] = v
                    else:
                        new_data[k] = v
                inner_data = new_data

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if isinstance(inner_kwargs, dict):
                new_kwargs = {}
                for k, v in inner_kwargs.items():
                    if isinstance(k, bytes):
                        new_kwargs[k.decode('utf-8')] = v
                    else:
                        new_kwargs[k] = v
                inner_kwargs = new_kwargs

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
        # --- Scenario A ---
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
    # Pre-emptively set up __builtin__ -> builtins mapping
    try:
        import builtins
        sys.modules['__builtin__'] = builtins
    except:
        pass
    main()