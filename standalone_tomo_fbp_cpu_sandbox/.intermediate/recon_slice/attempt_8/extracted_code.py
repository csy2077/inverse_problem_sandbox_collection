import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to ensure that the modules required by the pickled objects are importable.
# The pmat object likely comes from a tomography library. Let's try to make it available.
# First, let's try to add relevant paths.
run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)

# Also add parent directories that might contain the tomo modules
sandbox_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox'
if os.path.isdir(sandbox_dir):
    sys.path.insert(0, sandbox_dir)
    for d in os.listdir(sandbox_dir):
        full = os.path.join(sandbox_dir, d)
        if os.path.isdir(full):
            sys.path.insert(0, full)

# Try adding site-packages or other common paths
for candidate in [
    '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/lib',
    '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/lib',
]:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)


def scan_and_add_paths(base_dir, max_depth=3):
    """Recursively scan for directories containing Python modules and add to sys.path."""
    if max_depth <= 0 or not os.path.isdir(base_dir):
        return
    try:
        for entry in os.listdir(base_dir):
            full = os.path.join(base_dir, entry)
            if os.path.isdir(full):
                # Check if it contains __init__.py or .py files
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


scan_and_add_paths(run_code_dir, max_depth=3)

from agent_recon_slice import recon_slice
from verification_utils import recursive_check


class SafeUnpickler(pickle.Unpickler):
    """Custom unpickler that tries to handle missing modules gracefully."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError) as e:
            print(f"  Warning: Could not find {module}.{name}: {e}")
            # Try alternative module paths
            alternatives = [
                module.replace('tomopy.', ''),
                module.replace('astra.', ''),
                'builtins',
            ]
            for alt in alternatives:
                try:
                    return super().find_class(alt, name)
                except:
                    pass
            raise


def load_data(filepath):
    """Load a dill-serialized pickle file."""
    print(f"  Attempting to load: {filepath}")
    print(f"  File size: {os.path.getsize(filepath)} bytes")

    # Read raw bytes for inspection
    with open(filepath, 'rb') as f:
        raw = f.read()

    print(f"  First 40 bytes hex: {raw[:40].hex()}")
    print(f"  First 100 bytes repr: {repr(raw[:100])}")

    # The hex starts with 80037d... which is pickle protocol 3, dict marker
    # 80 = PROTO, 03 = version 3, 7d = EMPTY_DICT
    # This is a standard pickle protocol 3 file

    errors = []

    # Attempt 1: Direct dill.load
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print(f"  SUCCESS: dill.load returned type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"dill.load: {type(e).__name__}: {e}")
        print(f"  dill.load failed: {e}")
        traceback.print_exc()

    # Attempt 2: pickle.load
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"  SUCCESS: pickle.load returned type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"pickle.load: {type(e).__name__}: {e}")
        print(f"  pickle.load failed: {e}")

    # Attempt 3: Try loading with different pickle protocols by manually 
    # unpickling with dill.loads
    try:
        data = dill.loads(raw)
        print(f"  SUCCESS: dill.loads returned type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"dill.loads: {type(e).__name__}: {e}")
        print(f"  dill.loads failed: {e}")

    # Attempt 4: pickle.loads
    try:
        data = pickle.loads(raw)
        print(f"  SUCCESS: pickle.loads returned type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"pickle.loads: {type(e).__name__}: {e}")
        print(f"  pickle.loads failed: {e}")

    # Attempt 5: Try with restricted unpickler
    try:
        buf = io.BytesIO(raw)
        data = SafeUnpickler(buf).load()
        print(f"  SUCCESS: SafeUnpickler returned type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"SafeUnpickler: {type(e).__name__}: {e}")
        print(f"  SafeUnpickler failed: {e}")

    # Attempt 6: Manual partial unpickling - try to extract just the dict structure
    # The file is pickle protocol 3 starting with an empty dict
    # We can try to manually deserialize it by catching errors at specific points
    try:
        # Use pickletools to analyze
        import pickletools
        buf = io.BytesIO(raw)
        print("\n  Pickle disassembly (first 50 ops):")
        ops = []
        try:
            buf2 = io.BytesIO(raw)
            for i, (opcode, arg, pos) in enumerate(pickletools.genops(buf2)):
                ops.append((opcode, arg, pos))
                if i < 50:
                    arg_repr = repr(arg)[:80] if arg is not None else ''
                    print(f"    {pos:6d}: {opcode.name:20s} {arg_repr}")
                if i > 200:
                    break
        except Exception as e2:
            print(f"    pickletools error: {e2}")

        # Look for module references in the raw bytes
        import re
        modules_found = set()
        for match in re.finditer(rb'c([a-zA-Z0-9_.]+)\n([a-zA-Z0-9_]+)\n', raw):
            mod = match.group(1).decode('utf-8', errors='replace')
            name = match.group(2).decode('utf-8', errors='replace')
            modules_found.add((mod, name))
        for match in re.finditer(rb'\x8c([^\x00-\x1f]{3,50})', raw):
            s = match.group(1).decode('utf-8', errors='replace')
            if '.' in s:
                modules_found.add(('short_string', s))

        if modules_found:
            print(f"\n  Module references found in pickle:")
            for mod, name in sorted(modules_found):
                print(f"    {mod}.{name}" if mod != 'short_string' else f"    string: {name}")
    except Exception as e:
        errors.append(f"analysis: {e}")

    raise RuntimeError(f"All load methods failed:\n" + "\n".join(f"  - {e}" for e in errors))


def try_manual_reconstruction(filepath):
    """
    If dill/pickle can't load the file due to missing classes,
    try to manually reconstruct the data by intercepting the unpickling.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    # Look for what class is causing the issue
    import re

    # Find all class references in the pickle stream
    # Short binary string opcode \x8c followed by length byte
    classes = []
    i = 0
    while i < len(raw):
        if raw[i] == 0x8c:  # SHORT_BINUNICODE
            length = raw[i + 1]
            s = raw[i + 2:i + 2 + length].decode('utf-8', errors='replace')
            classes.append((i, s))
            i += 2 + length
        elif raw[i] == 0x8d:  # SHORT_BINUNICODE (2-byte length)
            length = int.from_bytes(raw[i + 1:i + 3], 'little')
            s = raw[i + 3:i + 3 + length].decode('utf-8', errors='replace')
            classes.append((i, s))
            i += 3 + length
        else:
            i += 1

    print(f"  Found {len(classes)} string references in pickle")
    for pos, s in classes[:100]:
        if '.' in s or s.startswith('_') or any(c.isupper() for c in s):
            print(f"    @{pos}: {s}")

    return classes


def create_mock_module(module_name, class_name):
    """Create a mock module with a mock class to allow unpickling."""
    import types

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

    # Create a mock class
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        def __getattr__(self, name):
            # Return a callable that returns None for any method call
            def mock_method(*args, **kwargs):
                return None
            return mock_method

    MockClass.__name__ = class_name
    MockClass.__qualname__ = class_name
    MockClass.__module__ = module_name

    setattr(current, class_name, MockClass)
    return MockClass


def load_with_mock_classes(filepath):
    """
    Try loading, and when it fails due to missing classes, mock them and retry.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    max_retries = 20
    for attempt in range(max_retries):
        try:
            data = dill.loads(raw)
            print(f"  SUCCESS on attempt {attempt + 1}")
            return data
        except (ModuleNotFoundError, ImportError) as e:
            msg = str(e)
            print(f"  Attempt {attempt + 1}: {msg}")

            # Extract module name from error
            # "No module named 'xxx'" or "No module named 'xxx.yyy'"
            import re
            match = re.search(r"No module named '([^']+)'", msg)
            if match:
                missing_module = match.group(1)
                print(f"    Creating mock module: {missing_module}")
                import types
                parts = missing_module.split('.')
                for i in range(len(parts)):
                    partial = '.'.join(parts[:i + 1])
                    if partial not in sys.modules:
                        mod = types.ModuleType(partial)
                        sys.modules[partial] = mod
                        if i > 0:
                            parent = sys.modules['.'.join(parts[:i])]
                            setattr(parent, parts[i], mod)
                continue
            else:
                raise
        except AttributeError as e:
            msg = str(e)
            print(f"  Attempt {attempt + 1}: AttributeError: {msg}")

            # "module 'xxx' has no attribute 'yyy'"
            import re
            match = re.search(r"module '([^']+)' has no attribute '([^']+)'", msg)
            if match:
                mod_name = match.group(1)
                attr_name = match.group(2)
                print(f"    Creating mock class: {mod_name}.{attr_name}")
                create_mock_module(mod_name, attr_name)
                continue

            # "type object 'X' has no attribute 'Y'"
            match = re.search(r"type object '([^']+)' has no attribute '([^']+)'", msg)
            if match:
                # Need to add attribute to the mock class
                class_name = match.group(1)
                attr_name = match.group(2)
                # Find the class in sys.modules
                for mod_name, mod in sys.modules.items():
                    if hasattr(mod, class_name):
                        cls = getattr(mod, class_name)
                        setattr(cls, attr_name, None)
                        break
                continue
            raise
        except Exception as e:
            print(f"  Attempt {attempt + 1}: {type(e).__name__}: {e}")
            raise

    raise RuntimeError(f"Failed to load after {max_retries} attempts")


def load_data_progressive(filepath):
    """
    Progressive loading strategy that handles missing dependencies.
    """
    print(f"\n  Loading: {filepath}")

    errors = []

    # Step 1: Try direct load
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print(f"  Direct dill.load succeeded: type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"direct dill: {e}")
        print(f"  Direct load failed: {type(e).__name__}: {e}")

    # Step 2: Try pickle
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"  Direct pickle.load succeeded: type={type(data).__name__}")
        return data
    except Exception as e:
        errors.append(f"direct pickle: {e}")
        print(f"  Direct pickle failed: {type(e).__name__}: {e}")

    # Step 3: Progressive loading with mock classes
    print(f"  Trying progressive loading with mock classes...")
    try:
        data = load_with_mock_classes(filepath)
        return data
    except Exception as e:
        errors.append(f"mock loading: {e}")
        print(f"  Mock loading failed: {e}")
        traceback.print_exc()

    # Step 4: Try to manually extract what we can
    print(f"  Trying manual extraction...")
    try:
        classes = try_manual_reconstruction(filepath)
    except Exception as e:
        errors.append(f"manual: {e}")

    raise RuntimeError(f"All loading strategies failed:\n" + "\n".join(f"  - {e}" for e in errors))


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_recon_slice.pkl'
    ]

    # Search for additional related files
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data'
    if os.path.isdir(std_data_dir):
        for f in sorted(os.listdir(std_data_dir)):
            full_path = os.path.join(std_data_dir, f)
            if full_path not in data_paths and 'recon_slice' in f and f.endswith('.pkl'):
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
        elif 'recon_slice' in basename and 'parent' not in basename:
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
    if inner_paths:
        for ip in inner_paths:
            print(f"Inner data file: {ip}")

    # --- Load outer data ---
    print(f"\n{'='*60}")
    print("PHASE 1: Loading outer data")
    print(f"{'='*60}")

    try:
        outer_data = load_data_progressive(outer_path)
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
                outer_kwargs = {(k.decode('utf-8') if isinstance(k, bytes) else k): v
                                for k, v in outer_kwargs.items()}
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
                for k, v in list(arg.__dict__.items())[:5]:
                    if isinstance(v, np.ndarray):
                        print(f"    .{k}: ndarray shape={v.shape}")
                    else:
                        print(f"    .{k}: {type(v).__name__}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}")

    for k, v in (outer_kwargs.items() if isinstance(outer_kwargs, dict) else []):
        if isinstance(v, np.ndarray):
            print(f"  kwarg[{k}]: ndarray shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  kwarg[{k}]: {type(v).__name__} = {str(v)[:100]}")

    # Check if pmat (3rd arg) has a working reconstruct method
    if len(outer_args) >= 3:
        pmat = outer_args[2]
        print(f"\n  pmat type: {type(pmat).__name__}")
        print(f"  pmat has reconstruct: {hasattr(pmat, 'reconstruct')}")
        if hasattr(pmat, 'reconstruct'):
            print(f"  pmat.reconstruct type: {type(pmat.reconstruct).__name__}")
            # Check if it's a mock
            if 'Mock' in type(pmat).__name__:
                print(f"  WARNING: pmat is a mock object - reconstruct will not work properly")

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
        print(f"Result sample: {agent_result.flat[:5]}")

    # --- Phase 3: Verify ---
    print(f"\n{'='*60}")
    print("PHASE 3: Verification")
    print(f"{'='*60}")

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from recon_slice, got {type(agent_result)}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}: {os.path.basename(inner_path)} ---")

            try:
                inner_data = load_data_progressive(inner_path)
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
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function, comparing output directly.")

        expected = outer_output
        result = agent_result

        print(f"Result type: {type(result)}")
        if isinstance(result, np.ndarray):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            print(f"Result sample (first 5): {result.flat[:5]}")
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Expected sample (first 5): {expected.flat[:5]}")

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