import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-inject __main__ attributes before loading pickle
import __main__
import functools
import inspect
import json

# Inject all symbols that the pickle might reference from __main__
_META_REGISTRY_ = set()
__main__._META_REGISTRY_ = _META_REGISTRY_
__main__.np = np
__main__.numpy = np
__main__._os_ = os
__main__._functools_ = functools
__main__._dill_ = dill
__main__._inspect_ = inspect
__main__._json_ = json
__main__._np_ = np

try:
    import torch
    __main__.torch = torch
    __main__._torch_ = torch
except ImportError:
    __main__._torch_ = None

def _analyze_obj_(obj):
    if __main__._torch_ and isinstance(obj, __main__._torch_.Tensor):
        return {'type': 'torch.Tensor', 'shape': list(obj.shape), 'dtype': str(obj.dtype), 'device': str(obj.device)}
    if isinstance(obj, np.ndarray):
        return {'type': 'numpy.ndarray', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return {'type': type(obj).__name__, 'length': len(obj), 'elements': [_analyze_obj_(item) for item in obj]}
    if hasattr(obj, '__dict__'):
        methods = []
        try:
            for m in dir(obj):
                if m.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, m)
                    if callable(attr):
                        methods.append(m)
                except Exception:
                    continue
        except Exception:
            pass
        return {'type': 'CustomObject', 'class_name': obj.__class__.__name__, 'public_methods': methods, 'attributes': list(obj.__dict__.keys())}
    try:
        val_str = str(obj)
    except:
        val_str = '<non-stringifiable>'
    return {'type': type(obj).__name__, 'value_sample': val_str}

__main__._analyze_obj_ = _analyze_obj_

def _record_io_decorator_(save_path='./'):
    def decorator(func, parent_function=None):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

__main__._record_io_decorator_ = _record_io_decorator_

def _data_capture_decorator_(func, parent_function=None):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

__main__._data_capture_decorator_ = _data_capture_decorator_

def _main_recon_slice(sinogram, method, pmat, parameters=None, pixel_size=1.0, offset=0):
    if type(offset) is float:
        offset = round(offset)
    if parameters is None:
        parameters = {}
    if 'iterations' in list(parameters.keys()):
        iterations = parameters['iterations']
        opts = {key: parameters[key] for key in parameters if key != 'iterations'}
    else:
        iterations = 1
        opts = parameters
    pixel_size = float(pixel_size)
    sinogram = sinogram / pixel_size
    if offset:
        sinogram = np.roll(sinogram, -offset, axis=1)
    rec = pmat.reconstruct(method, sinogram, iterations=iterations, extraOptions=opts)
    return rec

__main__.recon_slice = _main_recon_slice

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def load_pickle_file(filepath):
    """Load pickle file using dill, which is what was used to create it."""
    file_size = os.path.getsize(filepath)
    print(f"  File size: {file_size} bytes")

    # The pickle was created with dill. The error "Can't get attribute '__main__' 
    # on <module 'builtins'>" comes from dill trying to reconstruct references.
    # The error "function() argument 2 must be dict, not module" comes from
    # pickle's find_class being overridden incorrectly.
    # The "Ran out of input" with dill suggests encoding issues.
    
    # Since the file starts with \x80\x03 (protocol 3) and uses dill._dill functions,
    # we need dill's unpickler specifically.

    # Method 1: Standard dill load (default encoding)
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print("  Loaded with dill (default)")
        return data
    except Exception as e:
        print(f"  dill default failed: {e}")

    # Method 2: dill with explicit settings
    try:
        with open(filepath, 'rb') as f:
            data = dill.loads(f.read())
        print("  Loaded with dill.loads")
        return data
    except Exception as e:
        print(f"  dill.loads failed: {e}")

    # Method 3: dill Unpickler directly
    try:
        with open(filepath, 'rb') as f:
            unpickler = dill.Unpickler(f)
            data = unpickler.load()
        print("  Loaded with dill.Unpickler")
        return data
    except Exception as e:
        print(f"  dill.Unpickler failed: {e}")

    # Method 4: Read raw bytes, find the actual end of pickle stream
    # The file might have trailing data or multiple pickle streams
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        
        # Find the STOP opcode for protocol 3
        # In pickle, '.' (0x2e) is the STOP opcode
        # We need to find where the first complete pickle stream ends
        import io
        import pickletools
        
        # Use pickletools to find the end of the first pickle stream
        buf = io.BytesIO(raw)
        pos = 0
        try:
            for opcode, arg, p in pickletools.genops(buf):
                pos = buf.tell()
                if opcode.name == 'STOP':
                    break
        except Exception:
            pass
        
        if pos > 0 and pos <= len(raw):
            print(f"  Pickle stream ends at byte {pos} of {len(raw)}")
            truncated = raw[:pos]
            try:
                data = dill.loads(truncated)
                print("  Loaded truncated stream with dill")
                return data
            except Exception as e2:
                print(f"  Truncated dill failed: {e2}")
    except Exception as e:
        print(f"  Raw analysis failed: {e}")

    # Method 5: Try with different dill settings
    try:
        old_settings = dill.settings.copy() if hasattr(dill, 'settings') else {}
        if hasattr(dill, 'settings'):
            dill.settings['recurse'] = True
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        if hasattr(dill, 'settings'):
            dill.settings.update(old_settings)
        print("  Loaded with dill recurse=True")
        return data
    except Exception as e:
        print(f"  dill recurse failed: {e}")

    raise RuntimeError(f"Could not load {filepath} after all attempts")


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data'

    # Discover additional pkl files
    if os.path.isdir(std_data_dir):
        all_files = sorted(os.listdir(std_data_dir))
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        print(f"All pkl files in std_data directory: {pkl_files}")

        for f in pkl_files:
            if 'recon_slice' in f:
                full_path = os.path.join(std_data_dir, f)
                if full_path not in data_paths:
                    data_paths.append(full_path)

    print(f"Discovered pkl files: {[os.path.basename(p) for p in data_paths]}")

    # Classify paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if not os.path.exists(p):
            print(f"  WARNING: {p} does not exist")
            continue
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif basename in ('data_recon_slice.pkl', 'standard_data_recon_slice.pkl'):
            outer_path = p

    if outer_path is None:
        print("FAIL: No valid outer data file found.")
        sys.exit(1)

    print(f"\nOuter path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # --- Load outer data ---
    try:
        print(f"\nLoading outer data from: {outer_path}")
        outer_data = load_pickle_file(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Parse outer data
    if isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    elif isinstance(outer_data, (list, tuple)):
        if len(outer_data) >= 3:
            outer_args = outer_data[0] if isinstance(outer_data[0], (list, tuple)) else (outer_data[0],)
            outer_kwargs = outer_data[1] if isinstance(outer_data[1], dict) else {}
            outer_output = outer_data[2]
        else:
            print(f"FAIL: Unexpected data format (list/tuple of length {len(outer_data)})")
            sys.exit(1)
    else:
        print(f"FAIL: Unexpected data format: {type(outer_data)}")
        sys.exit(1)

    # Handle bytes keys from Python 2 pickle
    if isinstance(outer_kwargs, dict):
        new_kwargs = {}
        for k, v in outer_kwargs.items():
            if isinstance(k, bytes):
                new_kwargs[k.decode('utf-8')] = v
            else:
                new_kwargs[k] = v
        outer_kwargs = new_kwargs

    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    for i, arg in enumerate(outer_args):
        if hasattr(arg, 'shape'):
            print(f"  arg[{i}]: {type(arg).__name__}, shape={arg.shape}, dtype={getattr(arg, 'dtype', 'N/A')}")
        elif hasattr(arg, '__len__') and not isinstance(arg, str):
            print(f"  arg[{i}]: {type(arg).__name__}, len={len(arg)}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}, value={str(arg)[:200]}")

    # --- Determine scenario and execute ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\nScenario B: Factory/Closure pattern.")

        try:
            print("Executing recon_slice with outer args to get operator...")
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice raised: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable, got {type(agent_operator)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                inner_data = load_pickle_file(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if isinstance(inner_kwargs, dict):
                new_kw = {}
                for k, v in inner_kwargs.items():
                    if isinstance(k, bytes):
                        new_kw[k.decode('utf-8')] = v
                    else:
                        new_kw[k] = v
                inner_kwargs = new_kw

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {os.path.basename(inner_path)}: {msg}")
                all_passed = False
            else:
                print(f"PASS: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("\nScenario A: Simple function call.")

        try:
            print("Executing recon_slice...")
            result = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice raised: {e}")
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
            print(f"FAIL: recursive_check raised: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

        print("\nTEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()