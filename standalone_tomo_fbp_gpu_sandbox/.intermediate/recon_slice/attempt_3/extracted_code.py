import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import types

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-emptively set up __builtin__ -> builtins mapping
try:
    import builtins
    sys.modules['__builtin__'] = builtins
except:
    pass


def create_dummy_main_module():
    """Create a fake __main__ module with common classes that might have been pickled."""
    pass


class _MainModuleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles objects pickled from __main__."""

    def find_class(self, module, name):
        if module == '__builtin__':
            module = 'builtins'
        if module == '__main__':
            # Try to find the class in various places
            # First check if it's in builtins
            import builtins as _builtins
            if hasattr(_builtins, name):
                return getattr(_builtins, name)

            # Check if it's in the current global scope
            if name in globals():
                return globals()[name]

            # Try importing from common modules
            for mod_name in ['numpy', 'dill', 'types']:
                try:
                    mod = __import__(mod_name)
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except:
                    pass

            # Try to find it in sys.modules
            for mod_key, mod_val in sys.modules.items():
                if mod_val is not None and hasattr(mod_val, name):
                    try:
                        return getattr(mod_val, name)
                    except:
                        continue

            # As a last resort, create a dynamic class
            # This handles cases where a class was defined in __main__ during data generation
            dynamic_class = type(name, (object,), {
                '__module__': '__main__',
                '__reduce__': lambda self: (type(self), ()),
            })
            globals()[name] = dynamic_class
            return dynamic_class

        return super().find_class(module, name)


class _DillMainModuleUnpickler(dill.Unpickler):
    """Custom dill unpickler that handles objects pickled from __main__."""

    def find_class(self, module, name):
        if module == '__builtin__':
            module = 'builtins'
        if module == '__main__':
            import builtins as _builtins
            if hasattr(_builtins, name):
                return getattr(_builtins, name)
            if name in globals():
                return globals()[name]
            for mod_name in ['numpy', 'dill', 'types']:
                try:
                    mod = __import__(mod_name)
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except:
                    pass
            for mod_key, mod_val in sys.modules.items():
                if mod_val is not None and hasattr(mod_val, name):
                    try:
                        return getattr(mod_val, name)
                    except:
                        continue
            dynamic_class = type(name, (object,), {
                '__module__': '__main__',
            })
            globals()[name] = dynamic_class
            return dynamic_class
        return super().find_class(module, name)


def setup_main_module_for_unpickling(filepath):
    """
    Analyze the pickle file to find what classes from __main__ are needed,
    and pre-create them in the __main__ module and globals.
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        # Scan for __main__ references in the pickle stream
        # Look for patterns like '__main__\n<classname>\n' (protocol 0)
        # or the string '__main__' followed by class names
        import re

        # Find all occurrences of __main__ followed by potential class names
        # In pickle protocol 2+, class references are stored differently
        # Let's scan for string patterns
        main_refs = set()

        # Protocol 0/1/2 pattern: c__main__\nClassName\n
        for match in re.finditer(b'c__main__\n([^\n]+)\n', raw):
            class_name = match.group(1).decode('utf-8', errors='replace')
            main_refs.add(class_name)

        # Protocol 2+ pattern: \x8c\x08__main__  followed by class name
        # The format is: SHORT_BINUNICODE len __main__ SHORT_BINUNICODE len ClassName
        idx = 0
        while idx < len(raw):
            # Look for __main__ as a short binary string
            if raw[idx:idx+1] == b'\x8c':  # SHORT_BINUNICODE
                try:
                    strlen = raw[idx+1]
                    s = raw[idx+2:idx+2+strlen].decode('utf-8', errors='replace')
                    if s == '__main__':
                        # Next opcode should be the class name
                        next_idx = idx + 2 + strlen
                        if next_idx < len(raw) and raw[next_idx:next_idx+1] == b'\x8c':
                            name_len = raw[next_idx+1]
                            class_name = raw[next_idx+2:next_idx+2+name_len].decode('utf-8', errors='replace')
                            main_refs.add(class_name)
                except:
                    pass
            # Also check BINUNICODE (longer strings)
            elif raw[idx:idx+1] == b'\x8d':  # BINUNICODE
                try:
                    strlen = int.from_bytes(raw[idx+1:idx+5], 'little')
                    s = raw[idx+5:idx+5+strlen].decode('utf-8', errors='replace')
                    if s == '__main__':
                        next_idx = idx + 5 + strlen
                        if next_idx < len(raw):
                            if raw[next_idx:next_idx+1] == b'\x8c':
                                name_len = raw[next_idx+1]
                                class_name = raw[next_idx+2:next_idx+2+name_len].decode('utf-8', errors='replace')
                                main_refs.add(class_name)
                            elif raw[next_idx:next_idx+1] == b'\x8d':
                                name_len = int.from_bytes(raw[next_idx+1:next_idx+5], 'little')
                                class_name = raw[next_idx+5:next_idx+5+name_len].decode('utf-8', errors='replace')
                                main_refs.add(class_name)
                except:
                    pass
            # Check for X (SHORT_BINUNICODE in protocol 3+)
            elif raw[idx:idx+1] == b'X':
                try:
                    strlen = int.from_bytes(raw[idx+1:idx+5], 'little')
                    if strlen < 200:
                        s = raw[idx+5:idx+5+strlen].decode('utf-8', errors='replace')
                        if s == '__main__':
                            next_idx = idx + 5 + strlen
                            # Try to read next string
                            for opcode in [b'\x8c', b'X']:
                                if raw[next_idx:next_idx+1] == opcode:
                                    if opcode == b'\x8c':
                                        name_len = raw[next_idx+1]
                                        class_name = raw[next_idx+2:next_idx+2+name_len].decode('utf-8', errors='replace')
                                    else:
                                        name_len = int.from_bytes(raw[next_idx+1:next_idx+5], 'little')
                                        class_name = raw[next_idx+5:next_idx+5+name_len].decode('utf-8', errors='replace')
                                    main_refs.add(class_name)
                                    break
                except:
                    pass
            idx += 1

        print(f"Found __main__ class references: {main_refs}")

        # Now try to find these classes in the gen_data_code context or create dummies
        import __main__ as main_module

        for class_name in main_refs:
            if not hasattr(main_module, class_name) and class_name not in globals():
                # Try to find in loaded modules first
                found = False
                for mod_key, mod_val in sys.modules.items():
                    if mod_val is not None and hasattr(mod_val, class_name):
                        try:
                            cls = getattr(mod_val, class_name)
                            setattr(main_module, class_name, cls)
                            globals()[class_name] = cls
                            found = True
                            print(f"Found {class_name} in module {mod_key}")
                            break
                        except:
                            continue

                if not found:
                    # Create a flexible dynamic class that can accept any attributes
                    dynamic_class = type(class_name, (object,), {
                        '__module__': '__main__',
                        '__init__': lambda self, *a, **kw: None,
                        '__setstate__': lambda self, state: self.__dict__.update(state) if isinstance(state, dict) else None,
                        '__getattr__': lambda self, name: None,
                    })
                    setattr(main_module, class_name, dynamic_class)
                    globals()[class_name] = dynamic_class
                    print(f"Created dynamic class for {class_name}")

        return main_refs
    except Exception as e:
        print(f"Warning during main module setup: {e}")
        return set()


def try_load_data(filepath):
    """Try multiple approaches to load the data file."""
    if not os.path.exists(filepath):
        return None, f"File does not exist: {filepath}"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return None, f"File is empty (0 bytes): {filepath}"

    print(f"File size: {file_size} bytes")

    # First, setup __main__ module with needed classes
    setup_main_module_for_unpickling(filepath)

    with open(filepath, 'rb') as f:
        raw = f.read()

    errors = []

    # Attempt 1: Custom dill unpickler that handles __main__
    try:
        buf = io.BytesIO(raw)
        data = _DillMainModuleUnpickler(buf).load()
        print("Success with _DillMainModuleUnpickler")
        return data, None
    except Exception as e:
        errors.append(f"_DillMainModuleUnpickler failed: {e}")

    # Attempt 2: Custom pickle unpickler that handles __main__
    try:
        buf = io.BytesIO(raw)
        data = _MainModuleUnpickler(buf).load()
        print("Success with _MainModuleUnpickler")
        return data, None
    except Exception as e:
        errors.append(f"_MainModuleUnpickler failed: {e}")

    # Attempt 3: dill.loads
    try:
        data = dill.loads(raw)
        print("Success with dill.loads")
        return data, None
    except Exception as e:
        errors.append(f"dill.loads failed: {e}")

    # Attempt 4: pickle.loads
    try:
        data = pickle.loads(raw)
        print("Success with pickle.loads")
        return data, None
    except Exception as e:
        errors.append(f"pickle.loads failed: {e}")

    # Attempt 5: Try with different encodings for both custom unpicklers
    for enc in ['latin1', 'bytes', 'ASCII']:
        try:
            buf = io.BytesIO(raw)
            unpickler = _MainModuleUnpickler(buf)
            unpickler.encoding = enc
            data = unpickler.load()
            print(f"Success with _MainModuleUnpickler encoding={enc}")
            return data, None
        except Exception as e:
            errors.append(f"_MainModuleUnpickler enc={enc}: {e}")

        try:
            buf = io.BytesIO(raw)
            unpickler = _DillMainModuleUnpickler(buf)
            unpickler.encoding = enc
            data = unpickler.load()
            print(f"Success with _DillMainModuleUnpickler encoding={enc}")
            return data, None
        except Exception as e:
            errors.append(f"_DillMainModuleUnpickler enc={enc}: {e}")

    # Attempt 6: Try modifying __main__ to have the gen_data functions and re-load
    try:
        import __main__ as main_module

        # Add gen_data functions to __main__
        from agent_recon_slice import recon_slice as _recon_slice
        setattr(main_module, 'recon_slice', _recon_slice)

        # Also add numpy
        setattr(main_module, 'np', np)
        setattr(main_module, 'numpy', np)

        buf = io.BytesIO(raw)
        data = _DillMainModuleUnpickler(buf).load()
        print("Success after adding functions to __main__")
        return data, None
    except Exception as e:
        errors.append(f"After adding to __main__: {e}")

    # Attempt 7: Manual reconstruction - parse the pickle to understand structure
    try:
        buf = io.BytesIO(raw)

        class DebugUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                print(f"  find_class called: module={module}, name={name}")
                if module == '__builtin__':
                    module = 'builtins'
                if module == '__main__':
                    import builtins as _builtins
                    if hasattr(_builtins, name):
                        return getattr(_builtins, name)
                    # Try current module's globals
                    if name in globals():
                        return globals()[name]
                    # Create dynamic
                    dynamic = type(name, (object,), {
                        '__module__': '__main__',
                        '__setstate__': lambda self, state: self.__dict__.update(state) if isinstance(state, dict) else None,
                    })
                    globals()[name] = dynamic
                    import __main__ as mm
                    setattr(mm, name, dynamic)
                    return dynamic
                return super().find_class(module, name)

        data = DebugUnpickler(buf).load()
        print("Success with DebugUnpickler")
        return data, None
    except Exception as e:
        errors.append(f"DebugUnpickler failed: {e}")
        traceback.print_exc()

    # Attempt 8: Use dill's session loading capabilities
    try:
        # Temporarily redirect __main__
        import __main__ as main_module
        old_dict = main_module.__dict__.copy()

        # Add everything we might need
        main_module.__dict__['np'] = np
        main_module.__dict__['numpy'] = np

        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print("Success with dill.load after __main__ setup")
        return data, None
    except Exception as e:
        errors.append(f"dill.load with __main__ setup: {e}")

    # Attempt 9: pickletools analysis then custom load
    try:
        import pickletools
        buf = io.BytesIO(raw)
        # Just try to get some info
        ops = []
        try:
            for opcode, arg, pos in pickletools.genops(buf):
                ops.append((opcode.name, arg, pos))
                if len(ops) > 50:
                    break
        except:
            pass
        print(f"First pickle opcodes: {[(o[0], str(o[1])[:50]) for o in ops[:20]]}")
    except Exception as e:
        errors.append(f"pickletools analysis: {e}")

    return None, "All load attempts failed:\n" + "\n".join(errors)


def inspect_file_header(filepath):
    """Print the first few bytes for debugging."""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(64)
        print(f"File header (hex): {header[:32].hex()}")
        print(f"File header (repr): {repr(header[:32])}")
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


def fix_bytes_keys(d):
    """Recursively fix bytes keys in dicts (from Python 2 pickles)."""
    if not isinstance(d, dict):
        return d
    new_d = {}
    for k, v in d.items():
        new_key = k.decode('utf-8') if isinstance(k, bytes) else k
        new_d[new_key] = fix_bytes_keys(v) if isinstance(v, dict) else v
    return new_d


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

    # Before loading, scan what __main__ classes are needed and pre-populate them
    # by importing from the gen_data_code context
    try:
        import __main__ as main_module

        # The gen_data_code defines these in __main__ context.
        # We need to make them available when unpickling.
        # Key insight: the pickle references classes/functions from __main__
        # because the data was generated in a script run as __main__.

        # Import everything that might be needed
        from agent_recon_slice import recon_slice as _rs
        setattr(main_module, 'recon_slice', _rs)
        setattr(main_module, 'np', np)
        setattr(main_module, 'numpy', np)

        # Also try to import and set up any dependencies
        # The pmat object is likely a custom class - we need to handle it
    except Exception as e:
        print(f"Warning during __main__ setup: {e}")

    outer_data, load_err = try_load_data(outer_path)

    if outer_data is None:
        # Try a more aggressive approach: execute gen_data_code-like setup in __main__
        try:
            import __main__ as main_module

            # The gen_data code imports and defines several things in __main__
            # Let's try to replicate that environment
            exec_globals = main_module.__dict__

            # Add common imports
            exec_globals['os'] = os
            exec_globals['np'] = np
            exec_globals['numpy'] = np
            exec_globals['dill'] = dill

            # Now try loading again
            with open(outer_path, 'rb') as f:
                raw = f.read()

            buf = io.BytesIO(raw)
            outer_data = _DillMainModuleUnpickler(buf).load()
            print("Success after aggressive __main__ setup")
        except Exception as e:
            print(f"Aggressive approach also failed: {e}")
            traceback.print_exc()

    if outer_data is None:
        # Try yet another approach: load with dill after importing all agent modules
        try:
            # Try to find and import agent modules that might define needed classes
            agent_dir = os.path.dirname(os.path.abspath(__file__))
            for fname in os.listdir(agent_dir):
                if fname.startswith('agent_') and fname.endswith('.py'):
                    mod_name = fname[:-3]
                    try:
                        mod = __import__(mod_name)
                        # Add all public attributes to __main__
                        import __main__ as main_module
                        for attr_name in dir(mod):
                            if not attr_name.startswith('_'):
                                setattr(main_module, attr_name, getattr(mod, attr_name))
                    except Exception:
                        pass

            # Also try importing from the run_code directory
            run_code_dir = os.path.dirname(std_data_dir)
            if os.path.isdir(run_code_dir):
                sys.path.insert(0, run_code_dir)
                for fname in os.listdir(run_code_dir):
                    if fname.endswith('.py') and not fname.startswith('test_'):
                        mod_name = fname[:-3]
                        try:
                            mod = __import__(mod_name)
                            import __main__ as main_module
                            for attr_name in dir(mod):
                                if not attr_name.startswith('_'):
                                    setattr(main_module, attr_name, getattr(mod, attr_name))
                        except Exception:
                            pass

            with open(outer_path, 'rb') as f:
                raw = f.read()
            buf = io.BytesIO(raw)
            outer_data = _DillMainModuleUnpickler(buf).load()
            print("Success after importing all agent modules")
        except Exception as e:
            print(f"Module import approach also failed: {e}")
            traceback.print_exc()

    if outer_data is None:
        # Nuclear option: try to manually reconstruct with a very permissive unpickler
        try:
            with open(outer_path, 'rb') as f:
                raw = f.read()

            class NuclearUnpickler(pickle.Unpickler):
                """Unpickler that creates stub classes for anything from __main__."""

                def find_class(self, module, name):
                    if module == '__builtin__':
                        module = 'builtins'
                    if module == '__main__':
                        # First try real lookups
                        import builtins as _builtins
                        if hasattr(_builtins, name):
                            return getattr(_builtins, name)

                        # Try all sys.modules
                        for mod_key, mod_val in sorted(sys.modules.items()):
                            if mod_val is not None:
                                try:
                                    if hasattr(mod_val, name):
                                        obj = getattr(mod_val, name)
                                        print(f"  Found {name} in {mod_key}")
                                        return obj
                                except:
                                    continue

                        # Create a very flexible stub
                        print(f"  Creating stub for __main__.{name}")

                        class MetaStub(type):
                            def __instancecheck__(cls, instance):
                                return True

                        stub = MetaStub(name, (object,), {
                            '__module__': '__main__',
                            '__init__': lambda self, *a, **kw: None,
                            '__setstate__': lambda self, state: (
                                self.__dict__.update(state) if isinstance(state, dict)
                                else setattr(self, '__state__', state)
                            ),
                            '__getattr__': lambda self, attr: (
                                self.__dict__.get(attr, None)
                            ),
                            '__call__': lambda self, *a, **kw: None,
                            '__repr__': lambda self: f"<Stub {name}>",
                        })

                        import __main__ as mm
                        setattr(mm, name, stub)
                        globals()[name] = stub
                        return stub

                    return super().find_class(module, name)

            buf = io.BytesIO(raw)
            outer_data = NuclearUnpickler(buf).load()
            print("Success with NuclearUnpickler")
        except Exception as e:
            print(f"Nuclear approach failed: {e}")
            traceback.print_exc()

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

    outer_data = fix_bytes_keys(outer_data)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if isinstance(outer_kwargs, dict):
        outer_kwargs = fix_bytes_keys(outer_kwargs)

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

    # Now import the function
    from agent_recon_slice import recon_slice
    from verification_utils import recursive_check

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
                print(f"FAIL: Could not load inner data file: {inner_err}")
                sys.exit(1)

            if isinstance(inner_data, list) and len(inner_data) == 1:
                inner_data = inner_data[0]

            inner_data = fix_bytes_keys(inner_data)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if isinstance(inner_kwargs, dict):
                inner_kwargs = fix_bytes_keys(inner_kwargs)

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

    # Also try to add the run_code directory to path for any module imports
    run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code'
    if os.path.isdir(run_code_dir):
        sys.path.insert(0, run_code_dir)

    # Try to pre-import modules that might be needed for unpickling
    try:
        # Import all agent modules to make their classes available
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in sorted(os.listdir(agent_dir)):
            if fname.startswith('agent_') and fname.endswith('.py'):
                mod_name = fname[:-3]
                try:
                    mod = __import__(mod_name)
                    import __main__ as main_module
                    for attr_name in dir(mod):
                        if not attr_name.startswith('_'):
                            try:
                                setattr(main_module, attr_name, getattr(mod, attr_name))
                            except:
                                pass
                except Exception as e:
                    pass
    except:
        pass

    # Also try to import common tomography packages
    for pkg in ['astra', 'tomopy', 'tigre', 'pylops']:
        try:
            __import__(pkg)
        except:
            pass

    main()