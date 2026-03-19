import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import types
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)

sandbox_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox'
if os.path.isdir(sandbox_dir):
    sys.path.insert(0, sandbox_dir)
    for d in os.listdir(sandbox_dir):
        full = os.path.join(sandbox_dir, d)
        if os.path.isdir(full):
            sys.path.insert(0, full)

def scan_and_add_paths(base_dir, max_depth=4):
    if max_depth <= 0 or not os.path.isdir(base_dir):
        return
    try:
        for entry in os.listdir(base_dir):
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                has_py = False
                try:
                    for f in os.listdir(full):
                        if f.endswith('.py') or f == '__init__.py':
                            has_py = True
                            break
                except:
                    pass
                if has_py and full not in sys.path:
                    sys.path.insert(0, full)
                scan_and_add_paths(full, max_depth - 1)
    except:
        pass

scan_and_add_paths(run_code_dir, max_depth=4)
scan_and_add_paths(sandbox_dir, max_depth=4)

import builtins
sys.modules['__builtin__'] = builtins


def create_mock_module(module_name, class_name=None):
    parts = module_name.split('.')
    current = None
    for i, part in enumerate(parts):
        full_name = '.'.join(parts[:i + 1])
        if full_name not in sys.modules:
            mod = types.ModuleType(full_name)
            sys.modules[full_name] = mod
            if current is not None:
                setattr(current, part, mod)
            current = mod
        else:
            current = sys.modules[full_name]
    if class_name:
        if not hasattr(current, class_name):
            class MockClass:
                def __init__(self, *args, **kwargs):
                    self._mock_args = args
                    self._mock_kwargs = kwargs
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                    else:
                        self._state = state
                def __getattr__(self, name):
                    if name.startswith('_'):
                        raise AttributeError(name)
                    def mock_method(*args, **kwargs):
                        return None
                    return mock_method
                def __reduce__(self):
                    return (type(self), ())
            MockClass.__name__ = class_name
            MockClass.__qualname__ = class_name
            MockClass.__module__ = module_name
            setattr(current, class_name, MockClass)
            return MockClass
    return current


def patch_dill_for_function_globals():
    """
    Patch dill to handle the case where function globals contain a type object
    instead of a dict. The error 'function() argument 2 must be dict, not type'
    occurs when dill tries to reconstruct a function with incorrect globals.
    """
    # Save originals
    import dill._dill as _dill
    
    # Patch _create_function if it exists
    if hasattr(_dill, '_create_function'):
        _original_create_function = _dill._create_function
        def _patched_create_function(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
            # If fglobals is not a dict, replace it with a proper dict
            if not isinstance(fglobals, dict):
                print(f"  [Patch] Fixing fglobals: was {type(fglobals)}, replacing with empty dict + builtins")
                fglobals = {'__builtins__': builtins.__dict__}
            return _original_create_function(fcode, fglobals, fname, fdefaults, fclosure, fdict, fkwdefaults)
        _dill._create_function = _patched_create_function
    
    # Also patch types.FunctionType creation via _dill
    original_FunctionType = types.FunctionType
    
    class PatchedFunctionType:
        """Wrapper to intercept FunctionType construction with bad globals."""
        def __new__(cls, code, globals_dict, name=None, argdefs=None, closure=None):
            if not isinstance(globals_dict, dict):
                print(f"  [PatchedFunctionType] Fixing globals: was {type(globals_dict)}")
                globals_dict = {'__builtins__': builtins.__dict__}
            if name is not None and argdefs is not None and closure is not None:
                return original_FunctionType(code, globals_dict, name, argdefs, closure)
            elif name is not None and argdefs is not None:
                return original_FunctionType(code, globals_dict, name, argdefs)
            elif name is not None:
                return original_FunctionType(code, globals_dict, name)
            else:
                return original_FunctionType(code, globals_dict)
    
    return original_FunctionType


def load_data_with_function_fix(filepath):
    """
    Load pickle/dill data, handling the 'function() argument 2 must be dict, not type' error
    by monkey-patching the function reconstruction.
    """
    print(f"  Loading: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    fsize = os.path.getsize(filepath)
    print(f"  File size: {fsize} bytes")
    
    if fsize == 0:
        raise ValueError(f"File is empty: {filepath}")

    with open(filepath, 'rb') as f:
        raw = f.read()
    
    print(f"  Header bytes: {raw[:4].hex()}")
    
    errors = []

    # Strategy 0: Try raw dill first (maybe it works without patches)
    try:
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via dill.load: type={type(data).__name__}")
        return data
    except TypeError as e:
        if 'function() argument 2 must be dict' in str(e):
            print(f"  dill.load failed with function globals issue, applying patches...")
            errors.append(f"dill.load: {e}")
        else:
            errors.append(f"dill.load: {e}")
    except EOFError as e:
        errors.append(f"dill.load: EOFError: {e}")
        print(f"  dill.load failed with EOFError, trying other strategies...")
    except Exception as e:
        errors.append(f"dill.load: {type(e).__name__}: {e}")

    # Strategy 1: Patch dill._dill._create_function to fix the globals issue
    try:
        import dill._dill as _dill_module
        
        _orig_create_function = None
        if hasattr(_dill_module, '_create_function'):
            _orig_create_function = _dill_module._create_function
            
            def _fixed_create_function(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
                if not isinstance(fglobals, dict):
                    # Build a proper globals dict
                    new_globals = {'__builtins__': builtins.__dict__}
                    # Try to extract useful things from the type if possible
                    if isinstance(fglobals, type):
                        try:
                            for attr_name in dir(fglobals):
                                if not attr_name.startswith('__'):
                                    new_globals[attr_name] = getattr(fglobals, attr_name)
                        except:
                            pass
                    fglobals = new_globals
                
                # Ensure numpy is available in globals
                if 'np' not in fglobals and 'numpy' not in fglobals:
                    fglobals['np'] = np
                    fglobals['numpy'] = np
                
                # Build args properly
                args = [fcode, fglobals]
                if fname is not None:
                    args.append(fname)
                if fdefaults is not None:
                    args.append(fdefaults)
                if fclosure is not None:
                    args.append(fclosure)
                
                func = types.FunctionType(*args)
                if fdict is not None and isinstance(fdict, dict):
                    func.__dict__.update(fdict)
                if fkwdefaults is not None:
                    func.__kwdefaults__ = fkwdefaults
                return func
            
            _dill_module._create_function = _fixed_create_function
        
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via patched dill: type={type(data).__name__}")
        
        # Restore original
        if _orig_create_function is not None:
            _dill_module._create_function = _orig_create_function
        
        return data
    except Exception as e:
        errors.append(f"patched _create_function: {type(e).__name__}: {e}")
        print(f"  Patched _create_function failed: {e}")
        # Restore
        try:
            if _orig_create_function is not None:
                _dill_module._create_function = _orig_create_function
        except:
            pass

    # Strategy 2: More aggressive patching - intercept at pickle level
    try:
        import dill._dill as _dill_module
        
        _orig = None
        if hasattr(_dill_module, '_create_function'):
            _orig = _dill_module._create_function
        
        def _aggressive_create_function(*args, **kwargs):
            # Handle various calling conventions
            if len(args) >= 2:
                fcode = args[0]
                fglobals = args[1]
                if not isinstance(fglobals, dict):
                    fglobals = {'__builtins__': builtins.__dict__, 'np': np, 'numpy': np}
                    args = (fcode, fglobals) + args[2:]
            
            # Try with fixed args
            try:
                if len(args) == 2:
                    return types.FunctionType(args[0], args[1])
                elif len(args) == 3:
                    return types.FunctionType(args[0], args[1], args[2])
                elif len(args) == 4:
                    return types.FunctionType(args[0], args[1], args[2], args[3])
                elif len(args) >= 5:
                    func = types.FunctionType(args[0], args[1], args[2], 
                                             args[3] if args[3] is not None else None,
                                             args[4] if args[4] is not None else None)
                    if len(args) > 5 and args[5] and isinstance(args[5], dict):
                        func.__dict__.update(args[5])
                    if len(args) > 6 and args[6]:
                        func.__kwdefaults__ = args[6]
                    return func
                else:
                    return types.FunctionType(*args)
            except TypeError:
                # If closure is wrong type, try without it
                fcode = args[0]
                fglobals = args[1] if isinstance(args[1], dict) else {'__builtins__': builtins.__dict__, 'np': np}
                fname = args[2] if len(args) > 2 else None
                fdefaults = args[3] if len(args) > 3 else None
                
                if fname:
                    if fdefaults:
                        return types.FunctionType(fcode, fglobals, fname, fdefaults)
                    return types.FunctionType(fcode, fglobals, fname)
                return types.FunctionType(fcode, fglobals)
        
        _dill_module._create_function = _aggressive_create_function
        
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via aggressive patched dill: type={type(data).__name__}")
        
        if _orig is not None:
            _dill_module._create_function = _orig
        return data
    except Exception as e:
        errors.append(f"aggressive patch: {type(e).__name__}: {e}")
        print(f"  Aggressive patch failed: {e}")
        traceback.print_exc()
        try:
            if _orig is not None:
                _dill_module._create_function = _orig
        except:
            pass

    # Strategy 3: Custom unpickler that intercepts REDUCE operations
    try:
        class FunctionFixUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == '__builtin__':
                    module = 'builtins'
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, ImportError, AttributeError):
                    create_mock_module(module, name)
                    return super().find_class(module, name)

        # Monkey-patch types.FunctionType temporarily
        _real_FunctionType = types.FunctionType
        _real_new = _real_FunctionType.__new__ if hasattr(_real_FunctionType, '__new__') else None
        
        # We can't easily patch FunctionType, but we can patch at the dill level
        # Try using dill with recurse mode
        buf = io.BytesIO(raw)
        data = dill.load(buf, ignore=True)
        print(f"  SUCCESS via dill.load(ignore=True): type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"dill ignore: {type(e).__name__}: {e}")
        print(f"  dill.load(ignore=True) failed: {e}")

    # Strategy 4: Analyze the pickle stream and manually reconstruct
    try:
        import pickletools
        print("\n  === Pickle Stream Analysis ===")
        buf = io.BytesIO(raw)
        ops = list(pickletools.genops(buf))
        
        # Look for the structure: we need to find args, kwargs, output
        # The data should be a dict with keys 'func_name', 'args', 'kwargs', 'output'
        print(f"  Total ops: {len(ops)}")
        
        # Print first 50 ops for debugging
        for i, (opcode, arg, pos) in enumerate(ops[:50]):
            arg_repr = repr(arg)[:80] if arg is not None else ''
            print(f"    {pos:6d}: {opcode.name:20s} {arg_repr}")
        
        if len(ops) > 50:
            print(f"    ... ({len(ops) - 50} more ops)")
            # Print last 20
            for i, (opcode, arg, pos) in enumerate(ops[-20:]):
                arg_repr = repr(arg)[:80] if arg is not None else ''
                print(f"    {pos:6d}: {opcode.name:20s} {arg_repr}")
    except Exception as e:
        print(f"  Pickle analysis failed: {e}")

    # Strategy 5: Manually extract data by patching at the lowest level
    try:
        import dill._dill as _dill_module
        
        # Save all originals
        originals = {}
        for attr in dir(_dill_module):
            if attr.startswith('_create_') or attr.startswith('_load_'):
                originals[attr] = getattr(_dill_module, attr)
        
        # Patch _create_function with maximum tolerance
        def _ultra_safe_create_function(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
            if not isinstance(fglobals, dict):
                fglobals = {
                    '__builtins__': builtins.__dict__,
                    'np': np,
                    'numpy': np,
                    'round': round,
                    'float': float,
                    'int': int,
                    'list': list,
                    'dict': dict,
                    'type': type,
                    'isinstance': isinstance,
                }
                # Try to import common modules into globals
                for mod_name in ['os', 'sys', 'math', 'functools', 'itertools']:
                    try:
                        fglobals[mod_name] = __import__(mod_name)
                    except:
                        pass
            
            try:
                build_args = [fcode, fglobals]
                if fname is not None:
                    build_args.append(fname)
                    if fdefaults is not None:
                        build_args.append(fdefaults)
                        if fclosure is not None:
                            build_args.append(fclosure)
                    elif fclosure is not None:
                        build_args.append(None)  # fdefaults
                        build_args.append(fclosure)
                
                func = types.FunctionType(*build_args)
                
                if fdict is not None and isinstance(fdict, dict):
                    func.__dict__.update(fdict)
                if fkwdefaults is not None and isinstance(fkwdefaults, dict):
                    func.__kwdefaults__ = fkwdefaults
                return func
            except Exception as inner_e:
                # Last resort: create with minimal args
                try:
                    func = types.FunctionType(fcode, fglobals)
                    return func
                except:
                    # Return a dummy function
                    def dummy(*a, **kw):
                        return None
                    dummy.__name__ = fname or 'unknown'
                    return dummy
        
        _dill_module._create_function = _ultra_safe_create_function
        
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via ultra-safe patched dill: type={type(data).__name__}")
        
        # Restore
        for attr, orig in originals.items():
            setattr(_dill_module, attr, orig)
        
        return data
    except Exception as e:
        errors.append(f"ultra-safe patch: {type(e).__name__}: {e}")
        print(f"  Ultra-safe patch failed: {e}")
        traceback.print_exc()
        # Restore
        try:
            for attr, orig in originals.items():
                setattr(_dill_module, attr, orig)
        except:
            pass

    # Strategy 6: Try to load with different dill settings
    for setting in ['recurse', 'byref']:
        try:
            old_val = getattr(dill.settings, setting, None)
            setattr(dill.settings, setting, True)
            buf = io.BytesIO(raw)
            
            import dill._dill as _dm
            _orig_cf = getattr(_dm, '_create_function', None)
            
            def _setting_safe_cf(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
                if not isinstance(fglobals, dict):
                    fglobals = {'__builtins__': builtins.__dict__, 'np': np}
                build_args = [fcode, fglobals]
                if fname is not None:
                    build_args.append(fname)
                if fdefaults is not None:
                    build_args.append(fdefaults)
                elif fclosure is not None:
                    build_args.append(None)
                if fclosure is not None:
                    build_args.append(fclosure)
                try:
                    func = types.FunctionType(*build_args)
                except:
                    func = types.FunctionType(fcode, fglobals)
                if fdict and isinstance(fdict, dict):
                    func.__dict__.update(fdict)
                if fkwdefaults and isinstance(fkwdefaults, dict):
                    func.__kwdefaults__ = fkwdefaults
                return func
            
            _dm._create_function = _setting_safe_cf
            
            data = dill.load(buf)
            print(f"  SUCCESS via dill with {setting}=True: type={type(data).__name__}")
            
            if old_val is not None:
                setattr(dill.settings, setting, old_val)
            if _orig_cf is not None:
                _dm._create_function = _orig_cf
            return data
        except Exception as e:
            errors.append(f"dill {setting}: {type(e).__name__}: {e}")
            print(f"  dill {setting} failed: {e}")
            try:
                if old_val is not None:
                    setattr(dill.settings, setting, old_val)
                if _orig_cf is not None:
                    _dm._create_function = _orig_cf
            except:
                pass

    # Strategy 7: Direct binary analysis - find the dict structure
    # Look for string markers in the pickle data
    try:
        print("\n  === Binary Analysis ===")
        # Find string references
        string_refs = set()
        i = 0
        while i < len(raw):
            # SHORT_BINUNICODE: opcode 0x8c, followed by 1-byte length
            if raw[i] == 0x8c and i + 1 < len(raw):
                length = raw[i + 1]
                if i + 2 + length <= len(raw):
                    try:
                        s = raw[i + 2:i + 2 + length].decode('utf-8')
                        string_refs.add(s)
                    except:
                        pass
            # BINUNICODE: opcode 0x8d, followed by 4-byte length
            elif raw[i] == 0x8d and i + 4 < len(raw):
                length = int.from_bytes(raw[i + 1:i + 5], 'little')
                if length < 10000 and i + 5 + length <= len(raw):
                    try:
                        s = raw[i + 5:i + 5 + length].decode('utf-8')
                        string_refs.add(s)
                    except:
                        pass
            i += 1
        
        print(f"  Found {len(string_refs)} string references")
        for s in sorted(string_refs):
            print(f"    '{s}'")
    except Exception as e:
        print(f"  Binary analysis failed: {e}")

    # Strategy 8: Use pickletools.dis for full analysis
    try:
        import pickletools
        buf = io.BytesIO(raw)
        output = io.StringIO()
        pickletools.dis(buf, output, annotate=1)
        disasm = output.getvalue()
        # Print last 100 lines
        lines = disasm.split('\n')
        print(f"\n  === Pickle Disassembly (last 100 of {len(lines)} lines) ===")
        for line in lines[-100:]:
            print(f"    {line}")
    except Exception as e:
        print(f"  Disassembly failed: {e}")

    raise RuntimeError(
        f"All loading strategies failed for {filepath}:\n" +
        "\n".join(f"  - {e}" for e in errors)
    )


def load_data(filepath):
    """Main loader with all strategies."""
    print(f"  Loading: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    fsize = os.path.getsize(filepath)
    print(f"  File size: {fsize} bytes")
    if fsize == 0:
        raise ValueError(f"File is empty: {filepath}")

    # Read entire file
    with open(filepath, 'rb') as f:
        raw = f.read()

    print(f"  Header bytes: {raw[:4].hex()}")
    
    errors = []

    # Pre-patch dill before any loading attempt
    import dill._dill as _dill_module
    _orig_create_function = getattr(_dill_module, '_create_function', None)

    def _safe_create_function(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
        if not isinstance(fglobals, dict):
            # The gen_data_code has decorators that create closures
            # The globals should include numpy and other commonly used modules
            fglobals = {
                '__builtins__': builtins.__dict__,
                'np': np,
                'numpy': np,
                'round': round,
                'float': float,
                'int': int,
                'list': list,
                'dict': dict,
                'type': type,
                'isinstance': isinstance,
                'str': str,
                'len': len,
                'range': range,
                'print': print,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'callable': callable,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
            }
            # Try to import commonly needed modules
            for mod_name in ['os', 'sys', 'functools', 'inspect', 'json', 'dill', 'types']:
                try:
                    fglobals[mod_name] = __import__(mod_name)
                except:
                    pass
            # Add aliases used in gen_data_code
            try:
                fglobals['_os_'] = __import__('os')
                fglobals['_functools_'] = __import__('functools')
                fglobals['_dill_'] = __import__('dill')
                fglobals['_inspect_'] = __import__('inspect')
                fglobals['_json_'] = __import__('json')
                fglobals['_np_'] = np
                fglobals['_torch_'] = None
                fglobals['_META_REGISTRY_'] = set()
            except:
                pass

        build_args = [fcode, fglobals]
        if fname is not None:
            build_args.append(fname)
        if fdefaults is not None:
            if len(build_args) == 2:
                build_args.append(fname)
            build_args.append(fdefaults)
        if fclosure is not None:
            while len(build_args) < 4:
                build_args.append(None)
            build_args.append(fclosure)

        try:
            func = types.FunctionType(*build_args)
        except TypeError:
            # Try progressively simpler constructions
            try:
                func = types.FunctionType(fcode, fglobals, fname or '<unknown>')
            except:
                try:
                    func = types.FunctionType(fcode, fglobals)
                except:
                    def dummy(*a, **kw):
                        return None
                    dummy.__name__ = fname or 'unknown'
                    return dummy

        if fdict is not None and isinstance(fdict, dict):
            func.__dict__.update(fdict)
        if fkwdefaults is not None and isinstance(fkwdefaults, dict):
            func.__kwdefaults__ = fkwdefaults
        return func

    # Apply patch
    _dill_module._create_function = _safe_create_function

    # Strategy 1: dill.load with patch applied
    try:
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via patched dill.load: type={type(data).__name__}")
        # Restore
        if _orig_create_function is not None:
            _dill_module._create_function = _orig_create_function
        return data
    except EOFError as e:
        errors.append(f"patched dill.load: EOFError: {e}")
        print(f"  Patched dill.load failed with EOFError: {e}")
    except Exception as e:
        errors.append(f"patched dill.load: {type(e).__name__}: {e}")
        print(f"  Patched dill.load failed: {e}")
        traceback.print_exc()

    # Strategy 2: Handle potential multiple pickle objects in file
    # Sometimes the file may have been written with protocol issues
    try:
        # The EOFError might mean the file was truncated or has extra data
        # Try reading just the first pickle object
        buf = io.BytesIO(raw)
        
        # Use pickletools to find the STOP opcode
        import pickletools
        ops = list(pickletools.genops(buf))
        
        # Find the position after STOP
        stop_pos = None
        for opcode, arg, pos in ops:
            if opcode.name == 'STOP':
                stop_pos = pos + 1
                break
        
        if stop_pos and stop_pos < len(raw):
            print(f"  Found STOP at position {stop_pos - 1}, file has {len(raw)} bytes")
            # Try loading just up to STOP
            truncated = raw[:stop_pos]
            buf = io.BytesIO(truncated)
            data = dill.load(buf)
            print(f"  SUCCESS via truncated load: type={type(data).__name__}")
            if _orig_create_function is not None:
                _dill_module._create_function = _orig_create_function
            return data
        elif stop_pos is None:
            print(f"  No STOP opcode found - file may be genuinely truncated")
            # The file might have been written with an incomplete write
            # Look for the last STOP-like position
            for i in range(len(raw) - 1, -1, -1):
                if raw[i] == 0x2e:  # '.' is the STOP opcode
                    truncated = raw[:i + 1]
                    try:
                        buf = io.BytesIO(truncated)
                        data = dill.load(buf)
                        print(f"  SUCCESS via reverse-scan truncated load at {i}: type={type(data).__name__}")
                        if _orig_create_function is not None:
                            _dill_module._create_function = _orig_create_function
                        return data
                    except:
                        continue
    except Exception as e:
        errors.append(f"truncated load: {type(e).__name__}: {e}")
        print(f"  Truncated load failed: {e}")

    # Strategy 3: Try with pickle directly (patching won't help pickle, but try anyway)
    try:
        buf = io.BytesIO(raw)
        data = pickle.load(buf)
        print(f"  SUCCESS via pickle.load: type={type(data).__name__}")
        if _orig_create_function is not None:
            _dill_module._create_function = _orig_create_function
        return data
    except Exception as e:
        errors.append(f"pickle.load: {type(e).__name__}: {e}")

    # Strategy 4: If EOFError persists, the file might have been written in append mode
    # or with a different protocol. Try all pickle protocols.
    
    # Strategy 5: Custom reconstruction
    # If we can't unpickle, try to manually parse enough to get the dict
    try:
        print("\n  Attempting manual reconstruction from pickle opcodes...")
        import pickletools
        buf = io.BytesIO(raw)
        ops = list(pickletools.genops(buf))
        
        # The data structure should be:
        # {'func_name': 'recon_slice', 'args': (...), 'kwargs': {...}, 'output': ...}
        # Since the function (decorator wrapper) can't be unpickled, we need to find
        # the actual args/kwargs/output
        
        # Print all ops for debugging
        print(f"  Total pickle ops: {len(ops)}")
        for i, (opcode, arg, pos) in enumerate(ops):
            arg_repr = repr(arg)[:100] if arg is not None else ''
            print(f"    [{i:4d}] {pos:6d}: {opcode.name:20s} {arg_repr}")
    except Exception as e:
        print(f"  Manual reconstruction failed: {e}")

    # Restore original
    if _orig_create_function is not None:
        _dill_module._create_function = _orig_create_function

    # Strategy 6: If the file was saved by gen_data_code's _data_capture_decorator_,
    # the payload contains {'func_name', 'args', 'kwargs', 'output'} where
    # args and kwargs may contain the decorated function result (a wrapper function).
    # The key issue is that the wrapper function references `_data_capture_decorator_`
    # globals. Let's try to make those available.
    
    try:
        # Create the gen_data_code functions in a module
        gen_module = types.ModuleType('gen_data_helpers')
        
        exec("""
import os as _os_
import functools as _functools_
import dill as _dill_
import inspect as _inspect_
import json as _json_
import numpy as _np_
_torch_ = None

_META_REGISTRY_ = set()

def _analyze_obj_(obj):
    return {'type': type(obj).__name__}

def _record_io_decorator_(save_path='./'):
    def decorator(func, parent_function=None):
        @_functools_.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def _data_capture_decorator_(func, parent_function=None):
    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
""", gen_module.__dict__)
        
        sys.modules['gen_data_helpers'] = gen_module
        
        # Now add these to the safe globals
        def _safe_create_function_v2(fcode, fglobals, fname=None, fdefaults=None, fclosure=None, fdict=None, fkwdefaults=None):
            if not isinstance(fglobals, dict):
                fglobals = {
                    '__builtins__': builtins.__dict__,
                    'np': np,
                    'numpy': np,
                }
                # Add all gen_data_code symbols
                fglobals.update(gen_module.__dict__)
            else:
                # Even if it is a dict, ensure key symbols are present
                if 'np' not in fglobals:
                    fglobals['np'] = np
            
            build_args = [fcode, fglobals]
            if fname is not None:
                build_args.append(fname)
            if fdefaults is not None:
                while len(build_args) < 3:
                    build_args.append(None)
                build_args.append(fdefaults)
            if fclosure is not None:
                while len(build_args) < 4:
                    build_args.append(None)
                build_args.append(fclosure)
            
            try:
                func = types.FunctionType(*build_args)
            except:
                try:
                    func = types.FunctionType(fcode, fglobals)
                except:
                    def dummy(*a, **kw):
                        return None
                    return dummy
            
            if fdict and isinstance(fdict, dict):
                func.__dict__.update(fdict)
            if fkwdefaults and isinstance(fkwdefaults, dict):
                func.__kwdefaults__ = fkwdefaults
            return func
        
        _dill_module._create_function = _safe_create_function_v2
        
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print(f"  SUCCESS via gen_data patched dill: type={type(data).__name__}")
        if _orig_create_function is not None:
            _dill_module._create_function = _orig_create_function
        return data
    except Exception as e:
        errors.append(f"gen_data patch: {type(e).__name__}: {e}")
        print(f"  Gen data patch failed: {e}")
        traceback.print_exc()
        if _orig_create_function is not None:
            _dill_module._create_function = _orig_create_function

    # Strategy 7: If dill gives EOFError but pickle gives TypeError,
    # the file might need a specific dill version or have been corrupted.
    # Try to find if there's a backup or alternative file
    std_data_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    alt_names = [
        basename.replace('data_', 'standard_data_'),
        basename.replace('.pkl', '_backup.pkl'),
        basename.replace('.pkl', '.dill'),
    ]
    for alt in alt_names:
        alt_path = os.path.join(std_data_dir, alt)
        if os.path.exists(alt_path) and alt_path != filepath:
            print(f"  Found alternative file: {alt_path}")
            try:
                return load_data(alt_path)
            except:
                pass

    raise RuntimeError(
        f"All loading strategies failed for {filepath}:\n" +
        "\n".join(f"  - {e}" for e in errors)
    )


def find_data_files(data_paths):
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data'
    additional = []
    if os.path.isdir(std_data_dir):
        for f in sorted(os.listdir(std_data_dir)):
            full_path = os.path.join(std_data_dir, f)
            if 'recon_slice' in f and f.endswith('.pkl'):
                if full_path not in data_paths:
                    additional.append(full_path)
                    print(f"Discovered additional data file: {f}")
    return data_paths + additional


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    data_paths = find_data_files(data_paths)

    print("=== File Discovery ===")
    for p in data_paths:
        if os.path.exists(p):
            sz = os.path.getsize(p)
            print(f"  File: {os.path.basename(p)}, size: {sz} bytes")
        else:
            print(f"  File NOT FOUND: {p}")

    # Classify paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        if not os.path.exists(p):
            continue
        if os.path.getsize(p) == 0:
            continue
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif 'recon_slice' in basename and 'parent' not in basename:
            if outer_path is None:
                outer_path = p

    if outer_path is None:
        for p in data_paths:
            if os.path.exists(p) and os.path.getsize(p) > 0 and p not in inner_paths:
                outer_path = p
                break

    if outer_path is None:
        print("FAIL: No valid outer data file found.")
        sys.exit(1)

    print(f"\nOuter data file: {outer_path}")
    for ip in inner_paths:
        print(f"Inner data file: {ip}")

    # --- Load outer data ---
    print(f"\n{'='*60}")
    print("PHASE 1: Loading outer data")
    print(f"{'='*60}")

    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    if outer_data is None:
        print("FAIL: Loaded data is None")
        sys.exit(1)

    print(f"\nLoaded data type: {type(outer_data).__name__}")
    if isinstance(outer_data, dict):
        print(f"Keys: {list(outer_data.keys())}")

    # Extract payload
    if isinstance(outer_data, dict):
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)

        if isinstance(outer_kwargs, dict):
            kkeys = list(outer_kwargs.keys())
            if kkeys and isinstance(kkeys[0], bytes):
                outer_kwargs = {
                    (k.decode('utf-8') if isinstance(k, bytes) else k): v
                    for k, v in outer_kwargs.items()
                }
    else:
        print(f"FAIL: Loaded data is not a dict, it's {type(outer_data)}")
        sys.exit(1)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")

    # Debug args
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
        elif isinstance(arg, str):
            print(f"  arg[{i}]: str = '{arg}'")
        elif hasattr(arg, '__class__'):
            cls = arg.__class__
            cls_name = f"{cls.__module__}.{cls.__name__}" if hasattr(cls, '__module__') else cls.__name__
            print(f"  arg[{i}]: {cls_name}")
            if hasattr(arg, 'reconstruct'):
                print(f"    -> has 'reconstruct' method")
            if hasattr(arg, '__dict__'):
                for k, v in list(arg.__dict__.items())[:10]:
                    if isinstance(v, np.ndarray):
                        print(f"    .{k}: ndarray shape={v.shape}, dtype={v.dtype}")
                    else:
                        vstr = str(v)[:80]
                        print(f"    .{k}: {type(v).__name__} = {vstr}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__} = {str(arg)[:80]}")

    for k, v in (outer_kwargs.items() if isinstance(outer_kwargs, dict) else []):
        if isinstance(v, np.ndarray):
            print(f"  kwarg[{k}]: ndarray shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  kwarg[{k}]: {type(v).__name__} = {str(v)[:100]}")

    # Import the target function
    from agent_recon_slice import recon_slice
    from verification_utils import recursive_check

    # Execute recon_slice
    print(f"\n{'='*60}")
    print("PHASE 2: Executing recon_slice")
    print(f"{'='*60}")

    try:
        agent_result = recon_slice(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: recon_slice execution raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Result type: {type(agent_result)}")
    if isinstance(agent_result, np.ndarray):
        print(f"Result shape: {agent_result.shape}, dtype: {agent_result.dtype}")
        if agent_result.size > 0:
            print(f"Result sample (first 5): {agent_result.flat[:5]}")
            print(f"Result min={agent_result.min()}, max={agent_result.max()}, mean={agent_result.mean()}")

    # --- Phase 3: Verify ---
    print(f"\n{'='*60}")
    print("PHASE 3: Verification")
    print(f"{'='*60}")

    if inner_paths:
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from recon_slice, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")

            try:
                inner_data = load_data(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            if isinstance(inner_data, dict):
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            else:
                print(f"FAIL: Inner data is not a dict")
                sys.exit(1)

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
        print("Scenario A detected: Simple function, comparing output directly.")

        expected = outer_output
        result = agent_result

        print(f"Result type: {type(result)}")
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            if result.size > 0:
                print(f"Result sample (first 5): {result.flat[:5]}")
                print(f"Result min={result.min()}, max={result.max()}")
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            if expected.size > 0:
                print(f"Expected sample (first 5): {expected.flat[:5]}")
                print(f"Expected min={expected.min()}, max={expected.max()}")

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