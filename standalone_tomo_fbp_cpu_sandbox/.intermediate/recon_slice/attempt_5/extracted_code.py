import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import types
import importlib
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


def analyze_pickle_bytes(raw):
    """Analyze pickle byte stream to understand its structure."""
    print(f"  File size: {len(raw)} bytes")
    if len(raw) < 2:
        print("  File too small")
        return
    
    if raw[0] == 0x80:
        print(f"  Pickle protocol: {raw[1]}")
    
    # Count STOP opcodes
    stop_count = raw.count(b'.')
    print(f"  Number of '.' bytes (potential STOPs): {stop_count}")
    
    # Check for common patterns
    if b'builtins\n__main__\n' in raw:
        print("  Contains GLOBAL 'builtins.__main__' reference")
    if b'\x8c\x08builtins\x8c\x08__main__' in raw:
        print("  Contains SHORT_BINUNICODE 'builtins'+'__main__' reference")
    
    # Look for dill-specific markers
    if b'dill' in raw:
        print("  Contains 'dill' references")
    if b'_dill' in raw:
        print("  Contains '_dill' references")


def patch_pickle_bytes(raw):
    """
    Patch pickle bytes to fix the builtins.__main__ issue.
    The problem: dill serializes a reference to the __main__ module as
    builtins.__main__, but on deserialization builtins doesn't have __main__.
    
    We need to replace this with something that resolves to the __main__ module.
    """
    patched = raw
    
    # For protocol 2 (0x80 0x02) or protocol 3 (0x80 0x03):
    # The GLOBAL opcode is 'c' followed by module\nname\n
    # Pattern: c builtins\n__main__\n
    # We want to replace with: c dill\n_load_type\n followed by pushing '__main__' string
    # Actually simpler: we can use a custom module
    
    # For STACK_GLOBAL (protocol 4+):
    # Push 'builtins', push '__main__', then opcode 0x93
    # \x8c\x08builtins\x8c\x08__main__\x93
    
    # The simplest fix: create a module with __main__ attribute and redirect there
    helper_mod_name = '_recon_pkl_helper_'
    helper_mod = types.ModuleType(helper_mod_name)
    helper_mod.__main__ = sys.modules['__main__']
    sys.modules[helper_mod_name] = helper_mod
    
    # Replace GLOBAL opcode pattern: c builtins\n__main__\n
    old_global = b'cbuiltins\n__main__\n'
    new_global = b'c' + helper_mod_name.encode() + b'\n__main__\n'
    if old_global in patched:
        patched = patched.replace(old_global, new_global)
        print("  Patched GLOBAL opcode pattern")
    
    # Also try with 'c' followed by space variations
    old_global2 = b'c__builtin__\n__main__\n'
    new_global2 = b'c' + helper_mod_name.encode() + b'\n__main__\n'
    if old_global2 in patched:
        patched = patched.replace(old_global2, new_global2)
        print("  Patched __builtin__ GLOBAL opcode pattern")
    
    # Replace SHORT_BINUNICODE pattern for STACK_GLOBAL
    # \x8c + length_byte + string
    old_sbu = b'\x8c\x08builtins\x8c\x08__main__\x93'
    new_mod_bytes = helper_mod_name.encode()
    new_sbu = bytes([0x8c, len(new_mod_bytes)]) + new_mod_bytes + b'\x8c\x08__main__\x93'
    if old_sbu in patched:
        patched = patched.replace(old_sbu, new_sbu)
        print("  Patched SHORT_BINUNICODE + STACK_GLOBAL pattern")
    
    # Also handle BINUNICODE (opcode 0x8d) for longer strings
    # \x8d + 4-byte length (little endian) + string
    old_bu = b'\x8d' + struct.pack('<I', 8) + b'builtins' + b'\x8c\x08__main__\x93'
    new_bu = b'\x8d' + struct.pack('<I', len(new_mod_bytes)) + new_mod_bytes + b'\x8c\x08__main__\x93'
    if old_bu in patched:
        patched = patched.replace(old_bu, new_bu)
        print("  Patched BINUNICODE + STACK_GLOBAL pattern")
    
    return patched


def find_and_patch_all_main_refs(raw):
    """
    More thorough patching that finds all variations of __main__ module references.
    """
    helper_mod_name = '_recon_pkl_helper_'
    helper_mod = types.ModuleType(helper_mod_name)
    helper_mod.__main__ = sys.modules['__main__']
    sys.modules[helper_mod_name] = helper_mod
    
    patched = bytearray(raw)
    
    # Strategy: find all positions where 'builtins' is followed (possibly with opcodes between)
    # by '__main__' and the STACK_GLOBAL opcode 0x93
    
    # Pattern 1: SHORT_BINUNICODE 'builtins' + SHORT_BINUNICODE '__main__' + STACK_GLOBAL
    pattern1 = bytes([0x8c, 8]) + b'builtins' + bytes([0x8c, 8]) + b'__main__' + bytes([0x93])
    helper_bytes = helper_mod_name.encode()
    replace1 = bytes([0x8c, len(helper_bytes)]) + helper_bytes + bytes([0x8c, 8]) + b'__main__' + bytes([0x93])
    
    result = bytes(patched)
    if pattern1 in result:
        result = result.replace(pattern1, replace1)
        print(f"  Patched {result.count(replace1)} SHORT_BINUNICODE STACK_GLOBAL patterns")
    
    # Pattern 2: GLOBAL opcode 
    for prefix in [b'cbuiltins\n__main__\n', b'c__builtin__\n__main__\n']:
        replacement = b'c' + helper_bytes + b'\n__main__\n'
        if prefix in result:
            result = result.replace(prefix, replacement)
            print(f"  Patched GLOBAL pattern: {prefix[:30]}")
    
    return result


def robust_load(filepath):
    """Load a dill-serialized pickle file with comprehensive error handling."""
    import __main__
    
    # Inject necessary items into __main__
    __main__.recon_slice = recon_slice
    __main__.np = np
    __main__.numpy = np
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    print(f"\n  Analyzing pickle file: {os.path.basename(filepath)}")
    analyze_pickle_bytes(raw)
    
    errors = []
    
    # =========================================================================
    # Attempt 1: Plain dill.load
    # =========================================================================
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print("  Loaded with plain dill.load")
        return data
    except Exception as e:
        errors.append(f"plain dill: {e}")
    
    # =========================================================================
    # Attempt 2: Monkeypatch builtins.__main__ and use pickle
    # =========================================================================
    # The error "function() argument 2 must be dict, not module" suggests that
    # somewhere in the deserialization, a function object is being reconstructed
    # and __main__ (a module) is being passed where a dict (globals) is expected.
    # This happens because dill uses builtins.__main__ to get the __main__ module's
    # __dict__ for function reconstruction. When we set builtins.__main__ = module,
    # pickle tries to use the module directly instead of its __dict__.
    
    # So the RIGHT fix is to NOT set builtins.__main__ to the module, but instead
    # to handle this in a custom unpickler that knows what to do.
    
    # =========================================================================
    # Attempt 3: Custom unpickler with proper __main__ handling
    # =========================================================================
    try:
        class SmartUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # When dill serializes a function defined in __main__, it stores
                # a reference that needs builtins.__main__ to resolve.
                # We need to return the actual __main__ module.
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__':
                    module = 'builtins'
                
                # Try normal resolution
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError, ImportError):
                    # Try importing the module
                    try:
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
                    except Exception:
                        pass
                    # Check sys.modules
                    if name in sys.modules:
                        return sys.modules[name]
                    raise
        
        buf = io.BytesIO(raw)
        data = SmartUnpickler(buf).load()
        print("  Loaded with SmartUnpickler")
        return data
    except Exception as e:
        errors.append(f"SmartUnpickler: {e}")
    
    # =========================================================================
    # Attempt 4: The "Ran out of input" error from dill and 
    # "function() argument 2 must be dict, not module" from pickle
    # suggest the file was saved with dill and contains dill-specific opcodes
    # that standard pickle can't handle, but dill can't find enough data.
    # 
    # The "Ran out of input" from dill is suspicious for a 4MB file.
    # This might mean dill is reading past the pickle STOP opcode and trying
    # to read more. Let's check if there are multiple objects.
    # =========================================================================
    
    # Find all STOP opcode positions
    # In pickle, STOP is '.' (0x2E), but '.' can appear in data too.
    # We need to use pickletools to properly analyze
    try:
        import pickletools
        buf = io.BytesIO(raw)
        ops = []
        try:
            for opcode, arg, pos in pickletools.genops(buf):
                ops.append((opcode, arg, pos))
                if opcode.name == 'STOP':
                    break
        except Exception:
            pass
        
        if ops:
            last_op = ops[-1]
            stop_pos = last_op[2] + 1  # position after STOP
            print(f"  First pickle stream ends at byte {stop_pos} out of {len(raw)}")
            
            if stop_pos < len(raw):
                print(f"  There are {len(raw) - stop_pos} extra bytes after first STOP")
                # The file might have trailing data that confuses dill
                truncated = raw[:stop_pos]
                
                # Try loading just the first pickle stream
                try:
                    buf = io.BytesIO(truncated)
                    data = dill.load(buf)
                    print("  Loaded truncated stream with dill")
                    return data
                except Exception as e:
                    errors.append(f"truncated dill: {e}")
                
                try:
                    buf = io.BytesIO(truncated)
                    data = pickle.load(buf)
                    print("  Loaded truncated stream with pickle")
                    return data
                except Exception as e:
                    errors.append(f"truncated pickle: {e}")
    except Exception as e:
        errors.append(f"pickletools analysis: {e}")
    
    # =========================================================================
    # Attempt 5: The issue might be that dill's load reads extra data after the
    # pickle stream for its own metadata. Let's try dill.loads with exact bytes.
    # =========================================================================
    try:
        data = dill.loads(raw)
        print("  Loaded with dill.loads")
        return data
    except Exception as e:
        errors.append(f"dill.loads: {e}")
    
    # =========================================================================
    # Attempt 6: Byte-level patching then dill.load
    # =========================================================================
    try:
        patched = find_and_patch_all_main_refs(raw)
        buf = io.BytesIO(patched)
        data = dill.load(buf)
        print("  Loaded with patched bytes + dill")
        return data
    except Exception as e:
        errors.append(f"patched dill: {e}")
    
    try:
        patched = find_and_patch_all_main_refs(raw)
        buf = io.BytesIO(patched)
        data = pickle.load(buf)
        print("  Loaded with patched bytes + pickle")
        return data
    except Exception as e:
        errors.append(f"patched pickle: {e}")
    
    # =========================================================================
    # Attempt 7: dill with different settings
    # =========================================================================
    for recurse_val in [True, False]:
        for byref_val in [True, False]:
            try:
                dill.settings['recurse'] = recurse_val
                dill.settings['byref'] = byref_val
                buf = io.BytesIO(raw)
                data = dill.load(buf)
                print(f"  Loaded with dill (recurse={recurse_val}, byref={byref_val})")
                return data
            except Exception as e:
                errors.append(f"dill recurse={recurse_val} byref={byref_val}: {e}")
            finally:
                dill.settings['recurse'] = False
                dill.settings['byref'] = False
    
    # =========================================================================
    # Attempt 8: The file might be using protocol 3 and the "Ran out of input"
    # might be a red herring from dill trying to handle __main__ internally.
    # Let's try using dill._dill directly with Unpickler
    # =========================================================================
    try:
        if hasattr(dill, '_dill'):
            buf = io.BytesIO(raw)
            u = dill._dill.Unpickler(buf)
            data = u.load()
            print("  Loaded with dill._dill.Unpickler")
            return data
    except Exception as e:
        errors.append(f"dill._dill.Unpickler: {e}")
    
    # =========================================================================
    # Attempt 9: Maybe the file needs __main__ to have specific attributes 
    # that dill is looking for during deserialization. Let's add more.
    # =========================================================================
    try:
        # Add everything from the gen_data_code context
        import functools
        import inspect
        import json
        
        __main__.functools = functools
        __main__.inspect = inspect
        __main__.json = json
        __main__.os = os
        __main__.dill = dill
        __main__.sys = sys
        
        buf = io.BytesIO(raw)
        data = dill.load(buf)
        print("  Loaded with enriched __main__ + dill")
        return data
    except Exception as e:
        errors.append(f"enriched __main__ dill: {e}")
    
    # =========================================================================
    # Attempt 10: The "Ran out of input" could mean dill expects a specific
    # file position. Let's try seeking to 0 explicitly and using file object.
    # =========================================================================
    try:
        import tempfile
        
        # Write to a temp file to ensure clean file handle
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'rb') as f:
                data = dill.load(f)
            print("  Loaded from temp file with dill")
            return data
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"temp file dill: {e}")
    
    # =========================================================================
    # Attempt 11: Read the pickle opcodes manually to understand the structure
    # and reconstruct what we need
    # =========================================================================
    try:
        import pickletools
        buf = io.BytesIO(raw)
        # Disassemble first few hundred ops to understand structure
        dis_output = io.StringIO()
        try:
            pickletools.dis(buf, annotate=1, output=dis_output)
        except Exception:
            pass
        dis_text = dis_output.getvalue()
        # Print first 2000 chars of disassembly
        print(f"  Pickle disassembly (first 2000 chars):")
        print(dis_text[:2000])
    except Exception as e:
        errors.append(f"pickletools.dis: {e}")
    
    # =========================================================================
    # Attempt 12: If dill "Ran out of input" is the consistent error, maybe
    # the file was saved with a DIFFERENT version of dill that uses a different
    # format. Try loading with cloudpickle if available.
    # =========================================================================
    try:
        import cloudpickle
        buf = io.BytesIO(raw)
        data = cloudpickle.load(buf)
        print("  Loaded with cloudpickle")
        return data
    except ImportError:
        errors.append("cloudpickle not available")
    except Exception as e:
        errors.append(f"cloudpickle: {e}")
    
    # =========================================================================
    # Attempt 13: The "Ran out of input" error might be coming from dill trying
    # to read a second object after the first. This happens with some versions
    # of dill that wrap the data. Let's check dill version and try accordingly.
    # =========================================================================
    try:
        print(f"  dill version: {dill.__version__}")
    except:
        pass
    
    # =========================================================================
    # Attempt 14: Try with explicit protocol and fix_imports
    # =========================================================================
    try:
        buf = io.BytesIO(raw)
        data = pickle.load(buf, fix_imports=True, encoding='latin1')
        print("  Loaded with pickle fix_imports=True encoding=latin1")
        return data
    except Exception as e:
        errors.append(f"pickle fix_imports latin1: {e}")
    
    # =========================================================================
    # Attempt 15: Subclass dill.Unpickler and override the problematic dispatch
    # =========================================================================
    try:
        class FixedDillUnpickler(dill.Unpickler):
            def find_class(self, module, name):
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__':
                    module = 'builtins'
                return super().find_class(module, name)
        
        buf = io.BytesIO(raw)
        u = FixedDillUnpickler(buf)
        data = u.load()
        print("  Loaded with FixedDillUnpickler")
        return data
    except Exception as e:
        errors.append(f"FixedDillUnpickler: {e}")
    
    raise RuntimeError(f"All load methods failed. Errors:\n" + "\n".join(f"  - {e}" for e in errors))


def load_with_dill_compat(filepath):
    """
    Try to load by examining the exact dill deserialization issue.
    The 'Ran out of input' from dill on a 4MB file is very suspicious.
    This likely means dill's Unpickler has a different read strategy.
    """
    import __main__
    __main__.recon_slice = recon_slice
    __main__.np = np
    __main__.numpy = np
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # Check if this is actually a valid pickle by trying to parse opcodes
    import pickletools
    
    try:
        buf = io.BytesIO(raw)
        ops = list(pickletools.genops(buf))
        print(f"  Total pickle opcodes: {len(ops)}")
        if ops:
            last_op = ops[-1]
            print(f"  Last opcode: {last_op[0].name} at position {last_op[2]}")
            
            # Calculate the actual end of pickle data
            # The last opcode should be STOP
            if last_op[0].name == 'STOP':
                actual_end = last_op[2] + 1
                print(f"  Pickle data ends at byte {actual_end}, file has {len(raw)} bytes")
                
                if actual_end < len(raw):
                    # There's extra data - this is why dill says "Ran out of input"
                    # dill might be trying to read this extra data
                    # Let's try loading just the pickle portion
                    truncated = raw[:actual_end]
                    
                    # But we still need to handle the builtins.__main__ issue
                    # Let's check if we even have that pattern in the truncated data
                    
                    # First, let's try with dill on truncated data
                    try:
                        buf = io.BytesIO(truncated)
                        data = dill.load(buf)
                        return data
                    except EOFError:
                        # dill might need the extra data
                        pass
                    except Exception as e:
                        print(f"  Truncated dill failed: {e}")
                    
                    # Try with pickle on truncated data  
                    try:
                        buf = io.BytesIO(truncated)
                        data = pickle.load(buf)
                        return data
                    except Exception as e:
                        print(f"  Truncated pickle failed: {e}")
                    
                    # The extra data might be dill metadata
                    extra = raw[actual_end:]
                    print(f"  Extra data ({len(extra)} bytes): {extra[:50]}...")
                    
                    # Maybe dill expects to load pickle + then some extra info
                    # Let's look at what the extra data contains
                    if extra.startswith(b'\x80'):
                        print("  Extra data starts with pickle protocol marker")
                        # Multiple pickle objects in file
                        try:
                            buf = io.BytesIO(raw)
                            obj1 = pickle.load(buf)
                            obj2 = pickle.load(buf)
                            print(f"  Loaded 2 objects: {type(obj1)}, {type(obj2)}")
                            # Usually the first object is the data
                            if isinstance(obj1, dict) and 'args' in obj1:
                                return obj1
                            if isinstance(obj2, dict) and 'args' in obj2:
                                return obj2
                            return obj1
                        except Exception as e:
                            print(f"  Multi-object pickle failed: {e}")
                
                elif actual_end == len(raw):
                    print("  Pickle data covers entire file (expected)")
    except Exception as e:
        print(f"  pickletools.genops failed: {e}")
    
    raise RuntimeError("load_with_dill_compat failed")


def ultimate_load(filepath):
    """
    Final attempt: handle all known issues with dill deserialization.
    """
    import __main__
    __main__.recon_slice = recon_slice
    __main__.np = np
    __main__.numpy = np
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    errors = []
    
    # The key insight from the errors:
    # - dill says "Ran out of input" -> dill reads past the end
    # - pickle says "function() argument 2 must be dict, not module" -> 
    #   this means pickle IS loading the data but when it tries to reconstruct
    #   a function, it gets a module where it expects globals dict
    # - pickle says "Can't get attribute '__main__' on <module 'builtins'>" ->
    #   when builtins doesn't have __main__ patched
    
    # The "function() argument 2 must be dict, not module" error is THE key.
    # This means pickle is trying to create a function with:
    #   types.FunctionType(code, globals_dict, ...)
    # But instead of globals_dict, it's getting a module object.
    
    # In dill, when it serializes a function from __main__, it saves a reference
    # to __main__.__dict__ as the globals. But the serialization of __main__.__dict__
    # involves serializing __main__ as a module, and then getting its __dict__.
    
    # When dill deserializes, it uses its own _dill._create_function which knows
    # how to handle this. But when we use pickle directly, it doesn't know.
    
    # So we NEED dill to work. The "Ran out of input" from dill is the real problem.
    # Let's investigate WHY dill runs out of input on a 4MB file.
    
    # Possible cause: dill version mismatch. The file was saved with a different
    # version of dill that has a different serialization format.
    
    # Let's try: maybe the file was saved with Python 2's cPickle/pickle and
    # imported with dill just for the dump call. In that case, the internal
    # format should be standard pickle.
    
    # Actually, looking at gen_data_code more carefully:
    # _dill_.dump(payload, f) - this uses dill.dump
    # And payload is: {'func_name': ..., 'args': ..., 'kwargs': ..., 'output': ...}
    # where args/kwargs/output are detached (CPU tensors/numpy arrays/plain objects)
    
    # The payload dict itself is straightforward. The issue must be that one of
    # the objects IN the payload (args, kwargs, or output) contains something
    # that references __main__.
    
    # Looking at recon_slice signature:
    # recon_slice(sinogram, method, pmat, parameters=None, pixel_size=1.0, offset=0)
    # - sinogram: numpy array
    # - method: string
    # - pmat: some object with .reconstruct method
    # - parameters: dict
    # - pixel_size: float  
    # - offset: int/float
    
    # The pmat object is likely the problematic one - it's a custom object
    # that was defined in __main__ or has methods from __main__.
    
    # Strategy: We need dill to work. Let's figure out why it says "Ran out of input"
    
    # Check: maybe the file was written in binary mode but something went wrong
    # Let's verify the file has a complete pickle stream
    
    import pickletools
    
    try:
        buf = io.BytesIO(raw)
        ops = list(pickletools.genops(buf))
        last_op = ops[-1]
        pickle_end = last_op[2] + 1  # STOP is 1 byte
        print(f"  Pickle stream has {len(ops)} opcodes, ends at byte {pickle_end}")
        print(f"  File total: {len(raw)} bytes")
        print(f"  Trailing bytes: {len(raw) - pickle_end}")
    except Exception as e:
        print(f"  Cannot parse pickle opcodes: {e}")
        pickle_end = len(raw)
    
    # KEY INSIGHT: dill.load might be trying to do something AFTER pickle.load
    # In some versions, dill.load calls pickle.load and then tries to read more.
    # If the file ends exactly at the pickle STOP, dill's additional read fails.
    
    # Let's check: what does dill.load actually do?
    try:
        import inspect
        dill_load_source = inspect.getsource(dill.load)
        # Check if dill.load reads additional data
        if 'Unpickler' in dill_load_source:
            print("  dill.load uses Unpickler")
        # Print first few lines
        for line in dill_load_source.split('\n')[:20]:
            print(f"    {line}")
    except Exception as e:
        print(f"  Cannot inspect dill.load: {e}")
    
    # Let's try: modify the raw bytes to add padding/extra data that dill expects
    # Or better yet: let's look at what dill.Unpickler does differently
    
    # Attempt: Use dill.Unpickler directly
    try:
        buf = io.BytesIO(raw)
        u = dill.Unpickler(buf)
        data = u.load()
        print("  Loaded with dill.Unpickler directly")
        return data
    except EOFError as e:
        errors.append(f"dill.Unpickler EOFError: {e}")
        print(f"  dill.Unpickler failed with EOFError: {e}")
        # Check position
        print(f"  Buffer position at error: {buf.tell()}")
    except Exception as e:
        errors.append(f"dill.Unpickler: {e}")
        print(f"  dill.Unpickler failed: {e}")
    
    # Attempt: Add extra empty pickle after the main one
    # Some versions of dill try to load a second object
    try:
        # Add an empty dict pickle after the main data
        extra_pickle = dill.dumps({})
        padded = raw + extra_pickle
        buf = io.BytesIO(padded)
        data = dill.load(buf)
        print("  Loaded with padded data + dill")
        return data
    except Exception as e:
        errors.append(f"padded dill: {e}")
    
    # Attempt: Add None pickle  
    try:
        extra_pickle = dill.dumps(None)
        padded = raw + extra_pickle
        buf = io.BytesIO(padded)
        data = dill.load(buf)
        print("  Loaded with None-padded data + dill")
        return data
    except Exception as e:
        errors.append(f"None padded dill: {e}")
    
    # Attempt: Maybe file needs newline at end
    try:
        padded = raw + b'\n'
        buf = io.BytesIO(padded)
        data = dill.load(buf)
        print("  Loaded with newline-padded data + dill")
        return data
    except Exception as e:
        errors.append(f"newline padded: {e}")
    
    # Attempt: Maybe the file contains the data correctly but dill._dill._load
    # does something extra. Let's try to monkeypatch dill to not do the extra read.
    try:
        # Save original
        if hasattr(dill, '_dill') and hasattr(dill._dill, 'Unpickler'):
            orig_load = dill._dill.Unpickler.load
            
            def patched_load(self):
                try:
                    return pickle.Unpickler.load(self)
                except EOFError:
                    # If EOFError, re-raise as-is
                    raise
            
            # Actually, let's just try calling the parent class load
            buf = io.BytesIO(raw)
            u = dill.Unpickler(buf)
            # Call pickle.Unpickler.load directly
            data = pickle.Unpickler.load(u)
            print("  Loaded with pickle.Unpickler.load on dill.Unpickler instance")
            return data
    except Exception as e:
        errors.append(f"pickle.Unpickler.load on dill instance: {e}")
        print(f"  pickle.Unpickler.load on dill instance failed: {e}")
    
    # Attempt: The "function() argument 2 must be dict, not module" suggests
    # that the function's __globals__ is being set to __main__ module instead of 
    # __main__.__dict__. In dill, _create_function handles this by extracting __dict__
    # from the module. Let's make sure dill's _create_function is available.
    try:
        # Ensure dill's function creators are registered
        if hasattr(dill, '_dill'):
            # Force dill to register all its reducers
            pass
        
        # Try with a dill Unpickler that has all dispatch entries
        buf = io.BytesIO(raw)
        u = dill.Unpickler(buf)
        
        # Check what dispatch entries dill adds
        if hasattr(u, 'dispatch'):
            dill_dispatch_count = len(u.dispatch)
            pickle_dispatch_count = len(pickle.Unpickler.dispatch) if hasattr(pickle.Unpickler, 'dispatch') else 0
            print(f"  dill dispatch entries: {dill_dispatch_count}, pickle: {pickle_dispatch_count}")
        
        data = u.load()
        return data
    except Exception as e:
        errors.append(f"dill dispatch check: {e}")
    
    # =========================================================================
    # CRITICAL ATTEMPT: The issue is likely dill version incompatibility.
    # The file was saved with one version of dill and we're loading with another.
    # In newer versions of dill, the Unpickler might try to read a "header" or
    # "footer" that older versions didn't write.
    # =========================================================================
    
    # Let's check: does the file start with dill's marker or just standard pickle?
    if raw[0:1] == b'\x80':
        # Standard pickle protocol marker
        proto = raw[1]
        print(f"  File starts with standard pickle protocol {proto}")
    
    # Try loading with stock pickle + custom function type handler
    try:
        import types as types_mod
        
        class FunctionFixUnpickler(pickle.Unpickler):
            """
            Custom unpickler that handles dill's function serialization
            without needing dill's full deserialization machinery.
            """
            def find_class(self, module, name):
                if module == 'builtins' and name == '__main__':
                    # Return __main__'s __dict__ instead of the module
                    # Wait, find_class should return a class/callable, not a dict
                    # The issue is that dill stores the globals as a reference
                    # to the __main__ module, and then extracts __dict__ from it.
                    # pickle's REDUCE or BUILD then uses this.
                    return sys.modules['__main__']
                if module == '__builtin__' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__':
                    module = 'builtins'
                
                # Handle dill's internal functions
                if module.startswith('dill'):
                    try:
                        mod = importlib.import_module(module)
                        obj = getattr(mod, name)
                        return obj
                    except Exception:
                        pass
                
                return super().find_class(module, name)
        
        buf = io.BytesIO(raw)
        data = FunctionFixUnpickler(buf).load()
        print("  Loaded with FunctionFixUnpickler")
        return data
    except Exception as e:
        errors.append(f"FunctionFixUnpickler: {e}")
        print(f"  FunctionFixUnpickler error: {e}")
        traceback.print_exc()
    
    # =========================================================================
    # ATTEMPT: Maybe the "Ran out of input" is actually happening DURING
    # the pickle load, not after. This could mean the pickle data itself
    # is malformed or the opcodes reference data beyond the file.
    # Let's check by examining where the error occurs.
    # =========================================================================
    try:
        buf = io.BytesIO(raw)
        u = dill.Unpickler(buf)
        
        # Wrap the load to catch the exact position
        try:
            data = u.load()
            return data
        except EOFError:
            pos = buf.tell()
            print(f"  dill.Unpickler ran out of input at position {pos}/{len(raw)}")
            
            if pos >= len(raw):
                print("  Confirmed: dill read entire file and wanted more")
                # This means dill expects additional data after the pickle stream
                # Some versions of dill store additional metadata
                
                # Try: append a simple STOP opcode
                try:
                    padded = raw + b'.'
                    buf = io.BytesIO(padded)
                    data = dill.load(buf)
                    print("  Loaded with STOP-padded data")
                    return data
                except Exception as e2:
                    errors.append(f"STOP padded: {e2}")
                
                # Try: append an empty pickle (just protocol header + STOP)
                try:
                    empty_proto = bytes([0x80, raw[1]]) + b'.'
                    padded = raw + empty_proto  
                    buf = io.BytesIO(padded)
                    data = dill.load(buf)
                    print("  Loaded with empty-pickle-padded data")
                    return data
                except Exception as e2:
                    errors.append(f"empty pickle padded: {e2}")
                
                # Try: append N)R. (empty reduce)
                try:
                    padded = raw + b'\x80\x03N.'
                    buf = io.BytesIO(padded)
                    data = dill.load(buf)
                    print("  Loaded with None-padded data v2")
                    return data
                except Exception as e2:
                    errors.append(f"None padded v2: {e2}")
    except Exception as e:
        errors.append(f"position check: {e}")
    
    # =========================================================================
    # ATTEMPT: Check if file was saved with dill.dump(obj, file, protocol=X)
    # and the protocol matters
    # =========================================================================
    for proto in [0, 1, 2, 3, 4, 5]:
        try:
            # Create a fake second object with matching protocol
            second_obj = pickle.dumps(None, protocol=proto)
            padded = raw + second_obj
            buf = io.BytesIO(padded)
            data = dill.load(buf)
            print(f"  Loaded with protocol-{proto} None padding")
            return data
        except Exception as e:
            pass
    
    # =========================================================================
    # FINAL ATTEMPT: Read file in append mode to see if dill writes multiple times
    # =========================================================================
    try:
        # The gen_data_code shows: _dill_.dump(payload, f)
        # This is a single dump call. The file should contain exactly one object.
        # dill.load should read exactly one object.
        
        # The "Ran out of input" error in dill.load typically comes from
        # the Unpickler.load method when it calls self.read(n) and gets
        # fewer bytes than expected.
        
        # BUT: pickletools.genops successfully parsed the entire file and found STOP.
        # This means the pickle DATA is complete. dill must be reading BEYOND the
        # pickle stream.
        
        # In some versions of dill, the Unpickler.load method is overridden to
        # do additional reads. Let's check:
        
        if hasattr(dill, 'Unpickler'):
            u_class = dill.Unpickler
            if hasattr(u_class, 'load'):
                import inspect
                try:
                    src = inspect.getsource(u_class.load)
                    print(f"  dill.Unpickler.load source:")
                    for line in src.split('\n')[:30]:
                        print(f"    {line}")
                except Exception:
                    print("  Cannot get source of dill.Unpickler.load")
    except Exception as e:
        errors.append(f"source inspection: {e}")
    
    raise RuntimeError(f"ultimate_load failed. Errors:\n" + "\n".join(f"  - {e}" for e in errors))


def try_all_loaders(filepath):
    """Master function that tries all loading strategies."""
    import __main__
    __main__.recon_slice = recon_slice  
    __main__.np = np
    __main__.numpy = np
    
    loaders = [
        ("robust_load", robust_load),
        ("load_with_dill_compat", load_with_dill_compat),
        ("ultimate_load", ultimate_load),
    ]
    
    all_errors = []
    
    for name, loader in loaders:
        try:
            print(f"\n{'='*60}")
            print(f"Trying {name}...")
            print(f"{'='*60}")
            data = loader(filepath)
            print(f"SUCCESS with {name}")
            return data
        except Exception as e:
            all_errors.append(f"{name}: {e}")
            print(f"FAILED: {name}: {e}")
    
    # Last resort: try to read the pickle manually
    print(f"\n{'='*60}")
    print("Last resort: manual pickle reconstruction")
    print(f"{'='*60}")
    
    try:
        data = manual_reconstruct(filepath)
        return data
    except Exception as e:
        all_errors.append(f"manual_reconstruct: {e}")
    
    raise RuntimeError(f"ALL loaders failed:\n" + "\n".join(f"  - {e}" for e in all_errors))


def manual_reconstruct(filepath):
    """
    As a last resort, try to manually parse and reconstruct the data.
    We know the structure: {'func_name': str, 'args': tuple, 'kwargs': dict, 'output': ...}
    """
    import __main__
    import pickletools
    
    __main__.recon_slice = recon_slice
    __main__.np = np
    __main__.numpy = np
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # Strategy: intercept the function creation that fails and fix it
    
    original_function_type = types.FunctionType
    
    class FunctionProxy:
        """Proxy that wraps function creation to handle module->dict conversion."""
        def __new__(cls, code, globals_arg, *args, **kwargs):
            if isinstance(globals_arg, types.ModuleType):
                # Convert module to its __dict__
                globals_arg = globals_arg.__dict__
            return original_function_type(code, globals_arg, *args, **kwargs)
    
    # Monkeypatch types.FunctionType temporarily
    # Actually we can't do that easily. Let's try a different approach.
    
    # Override the REDUCE dispatch in pickle to intercept function creation
    class InterceptUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'builtins' and name == '__main__':
                return sys.modules['__main__']
            if module == '__builtin__' and name == '__main__':
                return sys.modules['__main__']
            if module == '__builtin__':
                module = 'builtins'
            
            # Handle dill's _create_function and similar
            if module.startswith('dill'):
                try:
                    mod = importlib.import_module(module)
                    return getattr(mod, name)
                except Exception:
                    pass
            
            obj = super().find_class(module, name)
            
            # If the resolved object is types.FunctionType, wrap it
            if obj is types.FunctionType:
                def safe_function_type(code, globals_dict, *args, **kwargs):
                    if isinstance(globals_dict, types.ModuleType):
                        globals_dict = globals_dict.__dict__
                    return types.FunctionType(code, globals_dict, *args, **kwargs)
                return safe_function_type
            
            return obj
    
    try:
        buf = io.BytesIO(raw)
        u = InterceptUnpickler(buf)
        data = u.load()
        print("  Loaded with InterceptUnpickler")
        return data
    except Exception as e:
        print(f"  InterceptUnpickler failed: {e}")
        traceback.print_exc()
    
    # Try another approach: replace types.FunctionType in builtins temporarily
    try:
        original_FunctionType = types.FunctionType
        
        class FunctionTypeWrapper:
            """Callable that creates FunctionType but handles module globals."""
            def __call__(self, *args, **kwargs):
                args = list(args)
                if len(args) >= 2 and isinstance(args[1], types.ModuleType):
                    args[1] = args[1].__dict__
                return original_FunctionType(*args, **kwargs)
            
            def __instancecheck__(cls, instance):
                return isinstance(instance, original_FunctionType)
        
        # We can't easily replace types.FunctionType, but we can try
        # to intercept at the dill level
        pass
    except Exception as e:
        print(f"  FunctionTypeWrapper approach failed: {e}")
    
    # Another approach: use dill but intercept the EOFError
    try:
        buf = io.BytesIO(raw)
        u = dill.Unpickler(buf)
        
        # Override the dispatch_table or persistent_load
        original_load = u.load
        
        # Actually, let's try to add extra data at the end that dill might need
        # dill stores __main__ session info sometimes
        
        # First, let's see what dill version was used to save
        # Check the pickle opcodes for dill-specific references
        buf2 = io.BytesIO(raw)
        ops = list(pickletools.genops(buf2))
        
        dill_refs = []
        for op, arg, pos in ops:
            if isinstance(arg, str) and 'dill' in arg:
                dill_refs.append((op.name, arg, pos))
        
        if dill_refs:
            print(f"  dill references in pickle:")
            for ref in dill_refs[:10]:
                print(f"    {ref}")
        
        # Check for __main__ references
        main_refs = []
        for op, arg, pos in ops:
            if isinstance(arg, str) and '__main__' in arg:
                main_refs.append((op.name, arg, pos))
        
        if main_refs:
            print(f"  __main__ references in pickle:")
            for ref in main_refs[:10]:
                print(f"    {ref}")
    except Exception as e:
        print(f"  Analysis failed: {e}")
    
    raise RuntimeError("manual_reconstruct failed")


def try_dill_with_session_padding(filepath):
    """
    Some versions of dill save session info after the main object.
    Try to append fake session data.
    """
    import __main__
    __main__.recon_slice = recon_slice
    __main__.np = np
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # Check what happens if we set __main__.__dict__ to have the expected items
    __main__.__builtins__ = __builtins__
    
    errors = []
    
    # Try appending various types of padding
    paddings = [
        dill.dumps(None),
        dill.dumps({}),
        dill.dumps(0),
        dill.dumps(''),
        dill.dumps(b''),
        dill.dumps([]),
        pickle.dumps(None, protocol=2),
        pickle.dumps(None, protocol=3),
        b'\x80\x02N.',  # Protocol 2 None
        b'\x80\x03N.',  # Protocol 3 None
        b'N.',          # Protocol 0 None
        b'.',           # Just STOP
        b'\x80\x02}.',  # Protocol 2 empty dict
        b'\x80\x03}.',  # Protocol 3 empty dict
    ]
    
    for i, padding in enumerate(paddings):
        try:
            padded = raw + padding
            buf = io.BytesIO(padded)
            data = dill.load(buf)
            print(f"  Success with padding #{i}")
            return data
        except Exception as e:
            errors.append(f"padding #{i}: {e}")
    
    raise RuntimeError(f"Session padding failed: {errors[:5]}")


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

    print(f"\nOuter data file: {outer_path} ({os.path.getsize(outer_path)} bytes)")
    if inner_paths:
        for ip in inner_paths:
            print(f"Inner data file: {ip} ({os.path.getsize(ip)} bytes)")

    # --- Load outer data ---
    print(f"\n{'='*60}")
    print("PHASE 1: Loading outer data")
    print(f"{'='*60}")
    
    outer_data = None
    
    # First, let's try the simplest approaches
    import __main__
    __main__.recon_slice = recon_slice
    __main__.np = np
    __main__.numpy = np
    
    # Read raw bytes for analysis
    with open(outer_path, 'rb') as f:
        raw = f.read()
    
    print(f"File size: {len(raw)} bytes")
    print(f"First bytes: {raw[:20]}")
    
    # Approach 1: InterceptUnpickler (handles function() argument 2 issue)
    try:
        class InterceptUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'builtins' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__' and name == '__main__':
                    return sys.modules['__main__']
                if module == '__builtin__':
                    module = 'builtins'
                if module.startswith('dill'):
                    try:
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
                    except Exception:
                        pass
                obj = super().find_class(module, name)
                if obj is types.FunctionType:
                    def safe_function_type(*args, **kwargs):
                        args = list(args)
                        if len(args) >= 2 and isinstance(args[1], types.ModuleType):
                            args[1] = args[1].__dict__
                        return types.FunctionType(*args, **kwargs)
                    return safe_function_type
                return obj
        
        buf = io.BytesIO(raw)
        outer_data = InterceptUnpickler(buf).load()
        print("Loaded with InterceptUnpickler")
    except Exception as e:
        print(f"InterceptUnpickler failed: {e}")
        traceback.print_exc()
    
    # Approach 2: dill with padding
    if outer_data is None:
        try:
            for padding in [b'\x80\x03N.', b'N.', b'.', dill.dumps(None)]:
                try:
                    padded = raw + padding
                    buf = io.BytesIO(padded)
                    outer_data = dill.load(buf)
                    print(f"Loaded with dill + padding")
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"dill with padding failed: {e}")
    
    # Approach 3: dill.loads
    if outer_data is None:
        try:
            outer_data = dill.loads(raw)
            print("Loaded with dill.loads")
        except Exception as e:
            print(f"dill.loads failed: {e}")
    
    # Approach 4: plain dill.load
    if outer_data is None:
        try:
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print("Loaded with dill.load")
        except Exception as e:
            print(f"dill.load failed: {e}")
    
    # Approach 5: Try comprehensive loading
    if outer_data is None:
        try:
            outer_data = try_all_loaders(outer_path)
        except Exception as e:
            print(f"All loaders failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    if outer_data is None:
        print("FAIL: Could not load outer data")
        sys.exit(1)

    # Validate loaded data
    if isinstance(outer_data, dict):
        # Handle bytes keys
        sample_keys = list(outer_data.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_data = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_data.items()
            }
    else:
        print(f"WARNING: outer_data is {type(outer_data)}, not dict")
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if isinstance(outer_kwargs, dict):
        sample_keys = list(outer_kwargs.keys())
        if sample_keys and isinstance(sample_keys[0], bytes):
            outer_kwargs = {
                (k.decode('utf-8') if isinstance(k, bytes) else k): v
                for k, v in outer_kwargs.items()
            }

    print(f"\nOuter data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Debug args
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
        elif isinstance(arg, str):
            print(f"  arg[{i}]: str = '{arg}'")
        elif hasattr(arg, '__class__'):
            print(f"  arg[{i}]: {arg.__class__.__name__}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}")

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

    # --- Phase 3: Determine scenario and verify ---
    print(f"\n{'='*60}")
    print("PHASE 3: Verification")
    print(f"{'='*60}")
    
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")
            
            inner_data = None
            # Try loading inner data with same approaches
            try:
                class InnerInterceptUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'builtins' and name == '__main__':
                            return sys.modules['__main__']
                        if module == '__builtin__' and name == '__main__':
                            return sys.modules['__main__']
                        if module == '__builtin__':
                            module = 'builtins'
                        if module.startswith('dill'):
                            try:
                                mod = importlib.import_module(module)
                                return getattr(mod, name)
                            except Exception:
                                pass
                        obj = super().find_class(module, name)
                        if obj is types.FunctionType:
                            def safe_ft(*args, **kwargs):
                                args = list(args)
                                if len(args) >= 2 and isinstance(args[1], types.ModuleType):
                                    args[1] = args[1].__dict__
                                return types.FunctionType(*args, **kwargs)
                            return safe_ft
                        return obj
                
                with open(inner_path, 'rb') as f:
                    inner_raw = f.read()
                buf = io.BytesIO(inner_raw)
                inner_data = InnerInterceptUnpickler(buf).load()
            except Exception:
                try:
                    with open(inner_path, 'rb') as f:
                        inner_raw = f.read()
                    for padding in [b'\x80\x03N.', dill.dumps(None)]:
                        try:
                            buf = io.BytesIO(inner_raw + padding)
                            inner_data = dill.load(buf)
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
            
            if inner_data is None:
                try:
                    inner_data = try_all_loaders(inner_path)
                except Exception as e:
                    print(f"FAIL: Could not load inner data: {e}")
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