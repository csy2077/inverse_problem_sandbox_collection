import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import types
import importlib

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import builtins

# The pickle file references '__main__' as an attribute lookup on builtins.
# This happens when dill serializes objects defined in __main__.
# We need to ensure that when unpickling tries to find '__main__', it gets the right module.

# Create a proper __builtin__ module reference
if '__builtin__' not in sys.modules:
    sys.modules['__builtin__'] = builtins

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def load_data(filepath):
    """Load a dill/pickle file with robust error handling."""
    errors = []
    file_size = os.path.getsize(filepath)

    # Check if file might be a standard pkl saved with dill from __main__ context
    # The error "Can't get attribute '__main__' on <module 'builtins'>" suggests
    # that dill serialized a reference to __main__ module itself.

    # Method 0: Try loading with dill in a way that handles __main__ references
    # by injecting necessary items into __main__
    try:
        import __main__
        # Make sure __main__ has the recon_slice function and other needed items
        if not hasattr(__main__, 'recon_slice'):
            __main__.recon_slice = recon_slice
        if not hasattr(__main__, 'np'):
            __main__.np = np

        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"dill.load (with __main__ injection): {e}")

    # Method 1: Try with dill.loads after reading all bytes
    try:
        import __main__
        if not hasattr(__main__, 'recon_slice'):
            __main__.recon_slice = recon_slice

        with open(filepath, 'rb') as f:
            raw = f.read()
        data = dill.loads(raw)
        return data
    except Exception as e:
        errors.append(f"dill.loads: {e}")

    # Method 2: Custom unpickler that resolves __main__ properly
    try:
        import __main__

        class MainPatchedUnpickler(dill.Unpickler):
            def find_class(self, module, name):
                if module == '__builtin__':
                    module = 'builtins'
                if name == '__main__':
                    return sys.modules['__main__']
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError):
                    # Try importing
                    try:
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
                    except Exception:
                        pass
                    # If looking for something in builtins that's actually a module
                    if module == 'builtins' and name in sys.modules:
                        return sys.modules[name]
                    raise

        with open(filepath, 'rb') as f:
            unpickler = MainPatchedUnpickler(f)
            data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"MainPatchedUnpickler (file): {e}")

    # Method 3: Read raw bytes and use MainPatchedUnpickler
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        class MainPatchedUnpickler2(dill.Unpickler):
            def find_class(self, module, name):
                if module == '__builtin__':
                    module = 'builtins'
                if name == '__main__':
                    return sys.modules['__main__']
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError):
                    if name in sys.modules:
                        return sys.modules[name]
                    raise

        unpickler = MainPatchedUnpickler2(io.BytesIO(raw))
        data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"MainPatchedUnpickler2 (bytes): {e}")

    # Method 4: Try with standard pickle and custom find_class
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        class StdPatchedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__builtin__':
                    module = 'builtins'
                if name == '__main__':
                    return sys.modules['__main__']
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError):
                    if name in sys.modules:
                        return sys.modules[name]
                    raise

        unpickler = StdPatchedUnpickler(io.BytesIO(raw))
        data = unpickler.load()
        return data
    except Exception as e:
        errors.append(f"StdPatchedUnpickler: {e}")

    # Method 5: The file might have been saved with multiple dumps or be corrupted.
    # Try reading only what we can.
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        # Try different offsets in case there's a header
        for offset in [0, 1, 2, 4, 8]:
            try:
                data = dill.loads(raw[offset:])
                return data
            except Exception:
                continue
    except Exception as e:
        errors.append(f"offset scan: {e}")

    # Method 6: The "Ran out of input" error from dill might mean file is
    # actually a different format or was saved differently. 
    # Let's check if it's maybe gzipped or otherwise compressed
    try:
        import gzip
        with gzip.open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"gzip+dill: {e}")

    # Method 7: Try with dill settings
    try:
        dill.settings['recurse'] = True
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"dill recurse: {e}")
    finally:
        dill.settings['recurse'] = False

    raise RuntimeError(
        f"All load methods failed for {filepath} (size={file_size}). Errors: {errors}"
    )


def load_data_with_fixes(filepath):
    """
    Attempt to load the pickle file by first examining its contents 
    and applying fixes.
    """
    # Read the raw bytes
    with open(filepath, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    print(f"  Raw file size: {file_size} bytes")
    print(f"  First 20 bytes: {raw[:20]}")
    print(f"  Last 20 bytes: {raw[-20:]}")

    # Check if the file has the pickle magic bytes
    # Protocol 2: \x80\x02, Protocol 3: \x80\x03, Protocol 4: \x80\x04, Protocol 5: \x80\x05
    if raw[0:1] == b'\x80':
        proto = raw[1]
        print(f"  Pickle protocol: {proto}")

    # Check if file ends with pickle STOP opcode '.'
    if raw[-1:] == b'.':
        print("  File ends with STOP opcode (good)")
    else:
        print(f"  File does NOT end with STOP opcode, last byte: {raw[-1:]}")
        # Maybe file is truncated or has trailing data
        # Find the last STOP opcode
        last_stop = raw.rfind(b'.')
        if last_stop > 0:
            print(f"  Last STOP opcode at position: {last_stop}")
            # Try loading up to last_stop + 1
            try:
                truncated = raw[:last_stop + 1]

                class TruncUnpickler(dill.Unpickler):
                    def find_class(self, module, name):
                        if module == '__builtin__':
                            module = 'builtins'
                        if name == '__main__':
                            return sys.modules['__main__']
                        if module == 'builtins' and name == '__main__':
                            return sys.modules['__main__']
                        try:
                            return super().find_class(module, name)
                        except (AttributeError, ModuleNotFoundError):
                            if name in sys.modules:
                                return sys.modules[name]
                            raise

                unpickler = TruncUnpickler(io.BytesIO(truncated))
                data = unpickler.load()
                return data
            except Exception as e:
                print(f"  Truncated load failed: {e}")

    # The "Ran out of input" error from dill suggests the file might have been
    # saved with a different version of dill or the file might contain
    # multiple serialized objects concatenated together.
    
    # Let's try to see if dill saved it with a session or something special
    # Try loading with different dill protocols
    
    # Try standard pickle with __main__ fix
    import __main__
    if not hasattr(__main__, 'recon_slice'):
        __main__.recon_slice = recon_slice
    if not hasattr(__main__, 'np'):
        __main__.np = np

    class FixedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle __builtin__ -> builtins
            if module == '__builtin__':
                module = 'builtins'
            # Handle the case where builtins.__main__ is requested
            if module == 'builtins' and name == '__main__':
                return sys.modules['__main__']
            # Handle dill internals
            if module.startswith('dill'):
                try:
                    mod = importlib.import_module(module)
                    return getattr(mod, name)
                except Exception:
                    pass
            return super().find_class(module, name)

    try:
        unpickler = FixedUnpickler(io.BytesIO(raw))
        data = unpickler.load()
        return data
    except Exception as e:
        print(f"  FixedUnpickler failed: {e}")
        traceback.print_exc()

    # As a last resort, try dill with session loading
    try:
        # Maybe it was saved with dill.dump_session or similar
        dill.load_module(filepath)
        # If this works, the data would be in __main__
        return None
    except Exception as e:
        print(f"  dill.load_module failed: {e}")

    raise RuntimeError(f"load_data_with_fixes failed for {filepath}")


def load_data_final(filepath):
    """Final attempt: investigate and load the pickle file."""
    import __main__
    import struct

    # Inject necessary attributes into __main__
    if not hasattr(__main__, 'recon_slice'):
        __main__.recon_slice = recon_slice
    if not hasattr(__main__, 'np'):
        __main__.np = np

    with open(filepath, 'rb') as f:
        raw = f.read()

    # The error "Ran out of input" from dill but "Can't get attribute '__main__'" from pickle
    # suggests dill is reading past the end somehow. Let's check if file has multiple
    # pickle streams or if dill wraps things differently.

    # Let's try to find all STOP opcodes
    stop_positions = []
    for i, b in enumerate(raw):
        if b == ord('.'):
            # Check if this could be a pickle STOP
            stop_positions.append(i)

    print(f"  Number of potential STOP positions: {len(stop_positions)}")
    if stop_positions:
        print(f"  Last few: {stop_positions[-5:]}")

    # The "Ran out of input" typically means dill expects MORE data.
    # This could happen if dill.dump was called but the file was saved with
    # standard open() and the data was written with a specific protocol.

    # Let's try: maybe the file was saved as 'standard_data_recon_slice.pkl'
    # but we're loading 'data_recon_slice.pkl'. Let's check the actual filename pattern.
    basename = os.path.basename(filepath)
    print(f"  Basename: {basename}")

    # Try with different dill Unpickler configurations
    errors = []

    # Attempt: Use dill._dill.Unpickler directly
    try:
        buf = io.BytesIO(raw)
        
        class DillFixedUnpickler(dill.Unpickler):
            dispatch = dill.Unpickler.dispatch.copy()

            def find_class(self, module, name):
                if module == '__builtin__':
                    module = 'builtins'
                if name == '__main__':
                    return sys.modules['__main__']
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                try:
                    return super().find_class(module, name)
                except Exception:
                    if name in sys.modules:
                        return sys.modules[name]
                    try:
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
                    except Exception:
                        pass
                    raise

        u = DillFixedUnpickler(buf)
        data = u.load()
        return data
    except Exception as e:
        errors.append(f"DillFixedUnpickler: {e}")
        traceback.print_exc()

    # Maybe the file actually has the data saved with 'standard_data_' prefix
    # Let's check the directory for the correct file
    data_dir = os.path.dirname(filepath)
    alt_name = 'standard_data_recon_slice.pkl'
    alt_path = os.path.join(data_dir, alt_name)
    if os.path.exists(alt_path) and alt_path != filepath:
        print(f"  Found alternative file: {alt_path}")
        try:
            with open(alt_path, 'rb') as f:
                data = dill.load(f)
            return data
        except Exception as e:
            errors.append(f"alt file dill: {e}")

    raise RuntimeError(f"load_data_final failed. Errors: {errors}")


def robust_load(filepath):
    """
    Ultimate robust loader that tries everything.
    """
    import __main__
    
    # Pre-populate __main__ with everything it might need
    if not hasattr(__main__, 'recon_slice'):
        __main__.recon_slice = recon_slice
    if not hasattr(__main__, 'np'):
        __main__.np = np
    if not hasattr(__main__, 'numpy'):
        __main__.numpy = np

    # Read raw bytes first
    with open(filepath, 'rb') as f:
        raw = f.read()

    total_size = len(raw)
    print(f"  File size: {total_size}")
    
    if total_size == 0:
        raise RuntimeError("Empty file")

    # Check protocol
    if raw[0] == 0x80:
        proto = raw[1]
        print(f"  Protocol: {proto}")
    
    # The key insight: "Ran out of input" from dill means dill's deserializer
    # expects more data than what's in the file. This can happen if:
    # 1. The file was truncated
    # 2. dill uses a different framing/header
    # 3. The file was written with a different serializer
    
    # "Can't get attribute '__main__'" from pickle means pickle found an
    # instruction to look up '__main__' as an attribute of builtins module.
    # This is how dill serializes references to the __main__ module itself.
    # In pickle opcodes, this looks like: GLOBAL 'builtins\n__main__\n'
    # or STACK_GLOBAL with ('builtins', '__main__')
    
    # Let's patch this at the byte level: replace the reference
    # The opcode sequence for STACK_GLOBAL might be:
    # SHORT_BINUNICODE 'builtins' + SHORT_BINUNICODE '__main__' + STACK_GLOBAL
    
    # Or for GLOBAL: c builtins\n__main__\n
    
    # Let's search for the pattern
    pattern1 = b'builtins\n__main__\n'
    pattern2 = b'\x8c\x08builtins\x8c\x08__main__'
    
    if pattern1 in raw:
        print("  Found GLOBAL pattern for builtins.__main__")
    if pattern2 in raw:
        print("  Found SHORT_BINUNICODE pattern for builtins.__main__")
    
    # Strategy: Use a custom Unpickler that properly handles __main__ references
    
    errors = []
    
    # Approach 1: Monkeypatch builtins to have __main__ attribute temporarily
    try:
        original = getattr(builtins, '__main__', None)
        has_original = hasattr(builtins, '__main__')
        builtins.__main__ = sys.modules['__main__']
        
        try:
            buf = io.BytesIO(raw)
            data = pickle.load(buf)
            return data
        finally:
            if has_original:
                builtins.__main__ = original
            else:
                try:
                    delattr(builtins, '__main__')
                except:
                    pass
    except Exception as e:
        errors.append(f"builtins monkeypatch + pickle: {e}")
    
    # Approach 2: Same but with dill
    try:
        original = getattr(builtins, '__main__', None)
        has_original = hasattr(builtins, '__main__')
        builtins.__main__ = sys.modules['__main__']
        
        try:
            buf = io.BytesIO(raw)
            data = dill.load(buf)
            return data
        finally:
            if has_original:
                builtins.__main__ = original
            else:
                try:
                    delattr(builtins, '__main__')
                except:
                    pass
    except Exception as e:
        errors.append(f"builtins monkeypatch + dill: {e}")

    # Approach 3: Byte-level patching - replace 'builtins\n__main__' with something valid
    # We can replace the GLOBAL opcode that loads builtins.__main__ with one that loads
    # from a module we control
    try:
        # Create a helper module that has __main__ as an attribute
        helper_mod = types.ModuleType('_pkl_helper_')
        helper_mod.__main__ = sys.modules['__main__']
        sys.modules['_pkl_helper_'] = helper_mod
        
        # Replace in raw bytes
        patched = raw.replace(b'builtins\n__main__\n', b'_pkl_helper_\n__main__\n')
        
        # Also handle SHORT_BINUNICODE format
        # \x8c\x08builtins -> need to replace with \x8c\x0c_pkl_helper_
        old_pattern = b'\x8c\x08builtins'
        new_name = b'_pkl_helper_'
        new_pattern = bytes([0x8c, len(new_name)]) + new_name
        
        # We need to be careful - only replace when followed by __main__
        # Let's do a more targeted replacement
        patched2 = raw
        search_seq = b'\x8c\x08builtins\x8c\x08__main__'
        replace_seq = b'\x8c\x0c_pkl_helper_\x8c\x08__main__'
        patched2 = patched2.replace(search_seq, replace_seq)
        
        for attempt_raw, desc in [(patched, 'GLOBAL patched'), (patched2, 'SHORT_BINUNICODE patched')]:
            try:
                buf = io.BytesIO(attempt_raw)
                data = pickle.load(buf)
                return data
            except Exception as e:
                errors.append(f"{desc} pickle: {e}")
            
            try:
                buf = io.BytesIO(attempt_raw)
                data = dill.load(buf)
                return data
            except Exception as e:
                errors.append(f"{desc} dill: {e}")
    except Exception as e:
        errors.append(f"byte patching: {e}")

    # Approach 4: More aggressive - find ALL occurrences and replace
    try:
        # Look for any reference pattern to builtins + __main__
        patched = raw
        
        # Pattern for protocol 2+ using SHORT_BINUNICODE (opcode 0x8c)
        # \x8c + len_byte + string
        # builtins = 8 chars -> \x8c\x08builtins
        # __main__ = 8 chars -> \x8c\x08__main__
        
        # But we also need STACK_GLOBAL opcode (0x93) after these two
        search = b'\x8c\x08builtins\x8c\x08__main__\x93'
        
        if search in patched:
            print(f"  Found exact STACK_GLOBAL pattern at positions: ", end="")
            pos = 0
            positions = []
            while True:
                pos = patched.find(search, pos)
                if pos == -1:
                    break
                positions.append(pos)
                pos += 1
            print(positions)
            
            # Replace with a GLOBAL opcode sequence that we can handle
            # Or better: just make builtins.__main__ work
            builtins.__main__ = sys.modules['__main__']
            try:
                buf = io.BytesIO(raw)
                data = pickle.load(buf)
                return data
            except Exception as e2:
                errors.append(f"exact pattern + monkeypatch: {e2}")
            finally:
                try:
                    delattr(builtins, '__main__')
                except:
                    pass
    except Exception as e:
        errors.append(f"pattern search: {e}")

    # Approach 5: Use dill with builtins patched AND handle "Ran out of input"
    # The "Ran out of input" might be because dill.load tries to load multiple objects
    try:
        builtins.__main__ = sys.modules['__main__']
        try:
            buf = io.BytesIO(raw)
            # Try loading just one object with pickle
            u = pickle.Unpickler(buf)
            data = u.load()
            return data
        except Exception as e:
            errors.append(f"pickle.Unpickler + monkeypatch: {e}")
        finally:
            try:
                delattr(builtins, '__main__')
            except:
                pass
    except Exception as e:
        errors.append(f"approach 5: {e}")

    # Approach 6: Check if pickle can load with encoding fixes
    try:
        builtins.__main__ = sys.modules['__main__']
        try:
            for enc in ['ASCII', 'latin1', 'bytes', 'utf-8']:
                try:
                    buf = io.BytesIO(raw)
                    u = pickle.Unpickler(buf, encoding=enc)
                    data = u.load()
                    return data
                except Exception as e:
                    errors.append(f"pickle enc={enc}: {e}")
        finally:
            try:
                delattr(builtins, '__main__')
            except:
                pass
    except Exception as e:
        errors.append(f"approach 6: {e}")

    raise RuntimeError(f"robust_load failed. Errors: {errors}")


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Search for additional related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        for f in sorted(os.listdir(std_data_dir)):
            full_path = os.path.join(std_data_dir, f)
            if full_path not in data_paths and 'recon_slice' in f:
                data_paths.append(full_path)
                print(f"Discovered additional data file: {f}")

    # Print file info
    for p in data_paths:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"File: {os.path.basename(p)}, size: {sz} bytes")
        else:
            print(f"File NOT FOUND: {p}")

    # Classify paths
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

    print(f"Outer data file: {outer_path} ({os.path.getsize(outer_path)} bytes)")

    # --- Phase 1: Load outer data and execute recon_slice ---
    print(f"Loading outer data from: {outer_path}")
    try:
        outer_data = robust_load(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file with robust_load: {e}")
        traceback.print_exc()
        
        # Last resort: try load_data_with_fixes
        try:
            outer_data = load_data_with_fixes(outer_path)
        except Exception as e2:
            print(f"FAIL: load_data_with_fixes also failed: {e2}")
            traceback.print_exc()
            
            # Ultimate last resort: try load_data
            try:
                outer_data = load_data(outer_path)
            except Exception as e3:
                print(f"FAIL: All loading methods exhausted: {e3}")
                sys.exit(1)

    # Handle bytes keys from Python 2 pickle loading
    if isinstance(outer_data, dict):
        sample_keys = list(outer_data.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_data = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_data.items()
            }

    if not isinstance(outer_data, dict):
        print(f"WARNING: outer_data is {type(outer_data)}, not dict")
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Handle bytes keys in kwargs
    if isinstance(outer_kwargs, dict):
        sample_keys = list(outer_kwargs.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_kwargs = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_kwargs.items()
            }

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Debug args
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
        elif isinstance(arg, str):
            print(f"  arg[{i}]: str = '{arg}'")
        else:
            print(f"  arg[{i}]: {type(arg).__name__} = {arg.__class__.__name__}")

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
            print(f"FAIL: Expected callable, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                inner_data = robust_load(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

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

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution error: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check error: {e}")
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
            print(f"FAIL: recursive_check error: {e}")
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