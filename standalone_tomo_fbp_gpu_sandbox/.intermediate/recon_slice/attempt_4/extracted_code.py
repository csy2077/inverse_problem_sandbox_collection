import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import types
import struct

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add run_code directory to path
run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)

# Pre-emptively set up __builtin__ -> builtins mapping
try:
    import builtins
    sys.modules['__builtin__'] = builtins
except:
    pass


def analyze_pickle_references(raw):
    """Scan pickle bytes for module.name references to understand what's needed."""
    import re
    refs = []

    # Protocol 0: c<module>\n<name>\n
    for match in re.finditer(b'c([^\n]+)\n([^\n]+)\n', raw):
        module = match.group(1).decode('utf-8', errors='replace')
        name = match.group(2).decode('utf-8', errors='replace')
        refs.append((module, name))

    # SHORT_BINUNICODE pairs (protocol 2+)
    idx = 0
    strings_found = []
    while idx < len(raw):
        byte = raw[idx]
        if byte == 0x8c:  # SHORT_BINUNICODE
            try:
                strlen = raw[idx + 1]
                s = raw[idx + 2:idx + 2 + strlen].decode('utf-8', errors='replace')
                strings_found.append((idx, s))
                idx += 2 + strlen
                continue
            except:
                pass
        elif byte == 0x8d:  # BINUNICODE
            try:
                strlen = struct.unpack('<I', raw[idx + 1:idx + 5])[0]
                if strlen < 1000:
                    s = raw[idx + 5:idx + 5 + strlen].decode('utf-8', errors='replace')
                    strings_found.append((idx, s))
                idx += 5 + strlen
                continue
            except:
                pass
        elif byte == 0x58:  # SHORT_BINUNICODE (X)
            try:
                strlen = struct.unpack('<I', raw[idx + 1:idx + 5])[0]
                if strlen < 1000:
                    s = raw[idx + 5:idx + 5 + strlen].decode('utf-8', errors='replace')
                    strings_found.append((idx, s))
                idx += 5 + strlen
                continue
            except:
                pass
        idx += 1

    # Find pairs that look like (module, name) for STACK_GLOBAL or similar
    for i in range(len(strings_found) - 1):
        mod_candidate = strings_found[i][1]
        name_candidate = strings_found[i + 1][1]
        if '.' not in mod_candidate and mod_candidate.replace('_', '').isalnum():
            refs.append((mod_candidate, name_candidate))

    return refs, strings_found


def setup_main_module_classes(raw):
    """Pre-populate __main__ with needed classes before unpickling."""
    import __main__ as main_module

    refs, strings = analyze_pickle_references(raw)

    main_classes_needed = set()
    for module, name in refs:
        if module == '__main__':
            main_classes_needed.add(name)
        elif module == 'builtins' and name == '__main__':
            # This is the problematic case: builtins.__main__ reference
            # This happens when the pickle tries to reference the __main__ module itself
            pass

    print(f"Classes needed from __main__: {main_classes_needed}")

    # Try to find these classes in loaded modules
    for class_name in main_classes_needed:
        if hasattr(main_module, class_name):
            continue

        found = False
        for mod_key, mod_val in sys.modules.items():
            if mod_val is not None and mod_key != '__main__':
                try:
                    if hasattr(mod_val, class_name):
                        obj = getattr(mod_val, class_name)
                        setattr(main_module, class_name, obj)
                        found = True
                        print(f"  Found {class_name} in {mod_key}")
                        break
                except:
                    continue

        if not found:
            # Create flexible stub
            stub = type(class_name, (object,), {
                '__module__': '__main__',
                '__init__': lambda self, *a, **kw: None,
                '__setstate__': lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else setattr(self, '_state_', state)
                ),
                '__call__': lambda self, *a, **kw: None,
                '__repr__': lambda self: f"<Stub {self.__class__.__name__}>",
            })
            setattr(main_module, class_name, stub)
            print(f"  Created stub for {class_name}")

    return main_classes_needed


class RobustUnpickler(pickle.Unpickler):
    """Unpickler that handles __main__ references and builtins.__main__ issues."""

    def find_class(self, module, name):
        # Fix __builtin__ -> builtins
        if module == '__builtin__':
            module = 'builtins'

        # Handle the specific case: builtins.__main__
        # This happens when pickle tries to load a reference to the __main__ module itself
        if module == 'builtins' and name == '__main__':
            import __main__
            return __main__

        # Handle __main__.X references
        if module == '__main__':
            import __main__ as main_module
            if hasattr(main_module, name):
                return getattr(main_module, name)

            # Try builtins
            import builtins as _builtins
            if hasattr(_builtins, name):
                return getattr(_builtins, name)

            # Search in all loaded modules
            for mod_key, mod_val in sys.modules.items():
                if mod_val is not None and mod_key != '__main__':
                    try:
                        if hasattr(mod_val, name):
                            obj = getattr(mod_val, name)
                            setattr(main_module, name, obj)
                            return obj
                    except:
                        continue

            # Create dynamic stub as last resort
            print(f"  Creating dynamic stub for __main__.{name}")
            stub = type(name, (object,), {
                '__module__': '__main__',
                '__init__': lambda self, *a, **kw: None,
                '__setstate__': lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else setattr(self, '_state_', state)
                ),
                '__getattr__': lambda self, attr_name: None,
                '__call__': lambda self, *a, **kw: None,
            })
            setattr(main_module, name, stub)
            return stub

        return super().find_class(module, name)


class RobustDillUnpickler(dill.Unpickler):
    """Dill unpickler that handles __main__ references and builtins.__main__ issues."""

    def find_class(self, module, name):
        if module == '__builtin__':
            module = 'builtins'

        if module == 'builtins' and name == '__main__':
            import __main__
            return __main__

        if module == '__main__':
            import __main__ as main_module
            if hasattr(main_module, name):
                return getattr(main_module, name)

            import builtins as _builtins
            if hasattr(_builtins, name):
                return getattr(_builtins, name)

            for mod_key, mod_val in sys.modules.items():
                if mod_val is not None and mod_key != '__main__':
                    try:
                        if hasattr(mod_val, name):
                            obj = getattr(mod_val, name)
                            setattr(main_module, name, obj)
                            return obj
                    except:
                        continue

            print(f"  Creating dynamic stub for __main__.{name}")
            stub = type(name, (object,), {
                '__module__': '__main__',
                '__init__': lambda self, *a, **kw: None,
                '__setstate__': lambda self, state: (
                    self.__dict__.update(state) if isinstance(state, dict) else setattr(self, '_state_', state)
                ),
                '__getattr__': lambda self, attr_name: None,
                '__call__': lambda self, *a, **kw: None,
            })
            setattr(main_module, name, stub)
            return stub

        return super().find_class(module, name)


def try_load_data(filepath):
    """Try multiple approaches to load the data file."""
    if not os.path.exists(filepath):
        return None, f"File does not exist: {filepath}"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return None, f"File is empty (0 bytes): {filepath}"

    print(f"File size: {file_size} bytes")

    with open(filepath, 'rb') as f:
        raw = f.read()

    # Inspect header
    print(f"File header (hex): {raw[:32].hex()}")
    if raw[:1] == b'\x80':
        print(f"Pickle protocol: {raw[1]}")

    # Check if the file might be truncated or have multiple concatenated pickles
    # dill sometimes writes in a way that has trailing data

    errors = []

    # Pre-populate __main__ with needed classes
    try:
        setup_main_module_classes(raw)
    except Exception as e:
        print(f"Warning during class setup: {e}")

    # Also ensure __main__ module itself is importable from builtins context
    import __main__ as main_module
    main_module.np = np
    main_module.numpy = np
    try:
        from agent_recon_slice import recon_slice as _rs
        main_module.recon_slice = _rs
    except:
        pass

    # Attempt 1: RobustUnpickler (pickle-based)
    try:
        buf = io.BytesIO(raw)
        data = RobustUnpickler(buf).load()
        print("Success with RobustUnpickler")
        return data, None
    except Exception as e:
        errors.append(f"RobustUnpickler: {e}")
        print(f"  RobustUnpickler failed: {e}")

    # Attempt 2: RobustDillUnpickler
    try:
        buf = io.BytesIO(raw)
        data = RobustDillUnpickler(buf).load()
        print("Success with RobustDillUnpickler")
        return data, None
    except Exception as e:
        errors.append(f"RobustDillUnpickler: {e}")
        print(f"  RobustDillUnpickler failed: {e}")

    # Attempt 3: Standard dill.loads
    try:
        data = dill.loads(raw)
        print("Success with dill.loads")
        return data, None
    except Exception as e:
        errors.append(f"dill.loads: {e}")
        print(f"  dill.loads failed: {e}")

    # Attempt 4: Standard pickle.loads
    try:
        data = pickle.loads(raw)
        print("Success with pickle.loads")
        return data, None
    except Exception as e:
        errors.append(f"pickle.loads: {e}")
        print(f"  pickle.loads failed: {e}")

    # Attempt 5: Try finding the correct pickle boundary
    # Sometimes the file has multiple pickle streams
    try:
        # Find STOP opcode (b'.')
        stop_positions = []
        for i, b in enumerate(raw):
            if b == ord('.'):
                # Check if this could be a pickle STOP
                stop_positions.append(i)

        for stop_pos in stop_positions[:10]:
            try:
                buf = io.BytesIO(raw[:stop_pos + 1])
                data = RobustUnpickler(buf).load()
                print(f"Success with RobustUnpickler at stop_pos={stop_pos}")
                return data, None
            except:
                pass
            try:
                buf = io.BytesIO(raw[:stop_pos + 1])
                data = RobustDillUnpickler(buf).load()
                print(f"Success with RobustDillUnpickler at stop_pos={stop_pos}")
                return data, None
            except:
                pass
    except Exception as e:
        errors.append(f"Boundary search: {e}")

    # Attempt 6: Try with encoding options
    for enc in ['latin1', 'bytes']:
        try:
            buf = io.BytesIO(raw)
            u = RobustUnpickler(buf)
            u.encoding = enc
            data = u.load()
            print(f"Success with RobustUnpickler encoding={enc}")
            return data, None
        except Exception as e:
            errors.append(f"RobustUnpickler enc={enc}: {e}")

    # Attempt 7: Monkey-patch dill to handle builtins.__main__
    try:
        original_find_class = dill.Unpickler.find_class

        def patched_find_class(self, module, name):
            if module == '__builtin__':
                module = 'builtins'
            if module == 'builtins' and name == '__main__':
                import __main__
                return __main__
            if module == '__main__':
                import __main__ as mm
                if hasattr(mm, name):
                    return getattr(mm, name)
            return original_find_class(self, module, name)

        dill.Unpickler.find_class = patched_find_class

        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print("Success with monkey-patched dill")
        dill.Unpickler.find_class = original_find_class
        return data, None
    except Exception as e:
        errors.append(f"Monkey-patched dill: {e}")
        try:
            dill.Unpickler.find_class = original_find_class
        except:
            pass

    # Attempt 8: Try using dill with recurse mode
    try:
        buf = io.BytesIO(raw)
        data = dill.load(buf, ignore=True)
        print("Success with dill.load(ignore=True)")
        return data, None
    except Exception as e:
        errors.append(f"dill.load(ignore=True): {e}")

    # Attempt 9: Low-level pickle reconstruction
    # The error is "Can't get attribute '__main__' on <module 'builtins'>"
    # This means there's a reference like builtins.__main__ in the pickle
    # We need to intercept this at the pickle opcode level
    try:
        import pickletools
        import copyreg

        # Manually patch the raw bytes to fix the reference
        # Replace 'builtins\n__main__' with a valid reference
        patched = raw.replace(b'builtins\n__main__\n', b'sys\nmodules\n')

        if patched != raw:
            print("Found and patched builtins.__main__ reference")
            buf = io.BytesIO(patched)
            try:
                data = RobustUnpickler(buf).load()
                print("Success with patched bytes + RobustUnpickler")
                return data, None
            except Exception as e2:
                errors.append(f"Patched bytes RobustUnpickler: {e2}")

        # Try another patch approach for protocol 2+
        # SHORT_BINUNICODE 'builtins' SHORT_BINUNICODE '__main__'
        builtins_bin = b'\x8c\x08builtins'
        main_bin = b'\x8c\x08__main__'

        if builtins_bin in raw and main_bin in raw:
            # Find positions where builtins is followed by __main__
            idx = 0
            while idx < len(raw):
                pos = raw.find(builtins_bin, idx)
                if pos == -1:
                    break
                after_builtins = pos + len(builtins_bin)
                if raw[after_builtins:after_builtins + len(main_bin)] == main_bin:
                    print(f"Found builtins/__main__ pair at position {pos}")
                    # Replace with __main__/__main__ reference that points to a dict
                    # Actually, replace the STACK_GLOBAL with loading sys.modules['__main__']
                    # For now, try replacing 'builtins' with a module that has __main__ attr
                    patched2 = raw[:pos] + b'\x8c\x03sys' + raw[after_builtins:]
                    buf = io.BytesIO(patched2)
                    try:
                        # Ensure sys has __main__ attr
                        sys.__main__ = main_module
                        data = RobustUnpickler(buf).load()
                        print("Success with binary-patched bytes")
                        return data, None
                    except Exception as e3:
                        errors.append(f"Binary patch attempt: {e3}")
                    break
                idx = pos + 1

    except Exception as e:
        errors.append(f"Low-level patching: {e}")

    # Attempt 10: Intercept at the lowest level by providing builtins.__main__
    try:
        import builtins as _builtins
        _builtins.__main__ = main_module

        buf = io.BytesIO(raw)
        data = pickle.loads(raw)
        print("Success after setting builtins.__main__")
        return data, None
    except Exception as e:
        errors.append(f"builtins.__main__ injection (pickle): {e}")

    try:
        buf = io.BytesIO(raw)
        data = dill.loads(raw)
        print("Success after setting builtins.__main__ (dill)")
        return data, None
    except Exception as e:
        errors.append(f"builtins.__main__ injection (dill): {e}")

    try:
        buf = io.BytesIO(raw)
        data = RobustUnpickler(buf).load()
        print("Success with RobustUnpickler after builtins.__main__")
        return data, None
    except Exception as e:
        errors.append(f"RobustUnpickler after builtins.__main__: {e}")

    # Attempt 11: Use pickletools to disassemble and understand structure
    try:
        import pickletools
        buf = io.BytesIO(raw)
        print("\n--- Pickle disassembly (first 50 ops) ---")
        ops = []
        try:
            for opcode, arg, pos in pickletools.genops(buf):
                ops.append((opcode.name, arg, pos))
                if len(ops) > 50:
                    break
        except Exception as e:
            print(f"  genops failed at op {len(ops)}: {e}")

        for op_name, arg, pos in ops[:30]:
            arg_str = repr(arg)[:80] if arg is not None else ''
            print(f"  {pos:6d}: {op_name:20s} {arg_str}")
        print("--- End disassembly ---\n")
    except Exception as e:
        print(f"Disassembly failed: {e}")

    return None, "All load attempts failed:\n" + "\n".join(errors)


def fix_bytes_keys(d):
    """Recursively fix bytes keys in dicts."""
    if not isinstance(d, dict):
        return d
    new_d = {}
    for k, v in d.items():
        new_key = k.decode('utf-8') if isinstance(k, bytes) else k
        new_d[new_key] = fix_bytes_keys(v) if isinstance(v, dict) else v
    return new_d


def find_all_data_files(base_dir, func_name):
    """Search for all related pkl files in the directory."""
    found = []
    if not os.path.isdir(base_dir):
        return found
    for fname in os.listdir(base_dir):
        if fname.endswith('.pkl') and func_name in fname:
            found.append(os.path.join(base_dir, fname))
    return sorted(found)


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Scan directory for related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        all_related = find_all_data_files(std_data_dir, 'recon_slice')
        print(f"All related pkl files found: {all_related}")
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

    print(f"\nOuter path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # --- Setup __main__ module before loading ---
    import __main__ as main_module

    # Add numpy and other common objects to __main__
    main_module.np = np
    main_module.numpy = np
    main_module.os = os

    # Import and add the target function
    try:
        from agent_recon_slice import recon_slice as _rs
        main_module.recon_slice = _rs
    except Exception as e:
        print(f"Warning: Could not import recon_slice: {e}")

    # Try to pre-import modules that might define classes needed during unpickling
    try:
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in sorted(os.listdir(agent_dir)):
            if fname.startswith('agent_') and fname.endswith('.py'):
                mod_name = fname[:-3]
                try:
                    mod = __import__(mod_name)
                    for attr_name in dir(mod):
                        if not attr_name.startswith('_'):
                            try:
                                setattr(main_module, attr_name, getattr(mod, attr_name))
                            except:
                                pass
                except:
                    pass
    except:
        pass

    # Try importing tomography-related packages
    for pkg in ['astra', 'tomopy', 'tigre', 'pylops', 'neutompy']:
        try:
            mod = __import__(pkg)
            setattr(main_module, pkg, mod)
        except:
            pass

    # CRITICAL: Set builtins.__main__ to the actual __main__ module
    # This fixes "Can't get attribute '__main__' on <module 'builtins'>"
    try:
        import builtins as _builtins
        _builtins.__main__ = main_module
        print("Set builtins.__main__ = __main__ module")
    except Exception as e:
        print(f"Warning setting builtins.__main__: {e}")

    # --- Load outer data ---
    print(f"\nLoading outer data from: {outer_path}")
    outer_data, load_err = try_load_data(outer_path)

    if outer_data is None:
        print(f"FAIL: Could not load outer data file.")
        print(f"Error: {load_err}")
        sys.exit(1)

    # Handle list wrapper
    if isinstance(outer_data, list) and not isinstance(outer_data, dict):
        if len(outer_data) == 1:
            outer_data = outer_data[0]
        else:
            for item in outer_data:
                if isinstance(item, dict) and ('args' in item or b'args' in item):
                    outer_data = item
                    break

    if not isinstance(outer_data, dict):
        print(f"WARN: outer_data is not a dict, it's {type(outer_data)}")
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
    # Pre-emptively set up builtins.__main__
    try:
        import builtins
        import __main__
        sys.modules['__builtin__'] = builtins
        builtins.__main__ = __main__
    except Exception as e:
        print(f"Warning during initial setup: {e}")

    main()