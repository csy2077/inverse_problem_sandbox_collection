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


def load_data(filepath):
    """Load a dill-serialized pickle file with robust error handling."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    errors = []

    # Attempt 1: Plain dill.load
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        if data is not None:
            print(f"  Loaded with plain dill.load, type={type(data)}")
            return data
        else:
            errors.append("plain dill: returned None")
    except Exception as e:
        errors.append(f"plain dill: {e}")

    # Attempt 2: Try multiple pickle objects in the file
    try:
        buf = io.BytesIO(raw)
        objects = []
        while buf.tell() < len(raw):
            try:
                obj = dill.load(buf)
                objects.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if objects:
            # Find the first dict-like object with 'args' key
            for obj in objects:
                if isinstance(obj, dict) and ('args' in obj or 'output' in obj):
                    print(f"  Loaded multi-object stream, found dict with keys={list(obj.keys())}")
                    return obj
            # Return last non-None object
            for obj in reversed(objects):
                if obj is not None:
                    print(f"  Loaded multi-object stream, returning type={type(obj)}")
                    return obj
        errors.append(f"multi-object: found {len(objects)} objects, none suitable")
    except Exception as e:
        errors.append(f"multi-object: {e}")

    # Attempt 3: Try with pickle directly
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if data is not None:
            print(f"  Loaded with pickle.load, type={type(data)}")
            return data
        errors.append("pickle: returned None")
    except Exception as e:
        errors.append(f"pickle: {e}")

    # Attempt 4: Scan raw bytes for pickle protocol markers and try loading from each
    try:
        positions = []
        for i in range(len(raw)):
            if raw[i:i+2] == b'\x80\x04' or raw[i:i+2] == b'\x80\x05' or raw[i:i+2] == b'\x80\x03' or raw[i:i+2] == b'\x80\x02':
                positions.append(i)

        for pos in positions:
            if pos == 0:
                continue
            try:
                buf = io.BytesIO(raw[pos:])
                obj = dill.load(buf)
                if obj is not None and isinstance(obj, dict) and ('args' in obj or 'output' in obj):
                    print(f"  Loaded from offset {pos}, type={type(obj)}")
                    return obj
            except Exception:
                continue

        # Try returning any non-None object from any position
        for pos in positions:
            if pos == 0:
                continue
            try:
                buf = io.BytesIO(raw[pos:])
                obj = dill.load(buf)
                if obj is not None:
                    print(f"  Loaded from offset {pos}, type={type(obj)}")
                    return obj
            except Exception:
                continue

        errors.append(f"offset scan: tried {len(positions)} positions, none worked")
    except Exception as e:
        errors.append(f"offset scan: {e}")

    # Attempt 5: The file might contain a None followed by the actual data
    try:
        buf = io.BytesIO(raw)
        first = dill.load(buf)  # This might be None
        remaining_pos = buf.tell()
        if remaining_pos < len(raw):
            remaining = raw[remaining_pos:]
            if len(remaining) > 10:
                buf2 = io.BytesIO(remaining)
                second = dill.load(buf2)
                if second is not None:
                    print(f"  Loaded second object after None, type={type(second)}")
                    return second
        errors.append("None-then-data: no second object")
    except Exception as e:
        errors.append(f"None-then-data: {e}")

    # Attempt 6: Read ALL objects from the stream
    try:
        buf = io.BytesIO(raw)
        all_objects = []
        while True:
            pos_before = buf.tell()
            if pos_before >= len(raw):
                break
            try:
                obj = dill.load(buf)
                all_objects.append((pos_before, obj))
            except EOFError:
                break
            except Exception:
                # Skip one byte and try again
                buf.seek(pos_before + 1)
                continue

        print(f"  Found {len(all_objects)} objects in stream")
        for i, (pos, obj) in enumerate(all_objects):
            t = type(obj).__name__
            if isinstance(obj, dict):
                print(f"    [{i}] at {pos}: dict keys={list(obj.keys())[:10]}")
            else:
                print(f"    [{i}] at {pos}: {t}")

        # Return the best candidate
        for pos, obj in all_objects:
            if isinstance(obj, dict) and 'args' in obj:
                return obj
        for pos, obj in reversed(all_objects):
            if obj is not None:
                return obj

        errors.append(f"exhaustive scan: {len(all_objects)} objects, none suitable")
    except Exception as e:
        errors.append(f"exhaustive scan: {e}")

    # Attempt 7: Look for STOP opcodes and try each segment
    try:
        stop_positions = [i for i in range(len(raw)) if raw[i:i+1] == b'.']
        print(f"  Found {len(stop_positions)} potential STOP opcodes")

        for stop_pos in stop_positions:
            segment = raw[:stop_pos + 1]
            try:
                obj = dill.loads(segment)
                if obj is not None and isinstance(obj, dict):
                    print(f"  Loaded segment ending at {stop_pos}, type={type(obj)}")
                    return obj
            except Exception:
                continue

            # Try from various protocol headers within this segment
            for start in range(len(segment)):
                if segment[start:start+1] == b'\x80':
                    try:
                        obj = dill.loads(segment[start:])
                        if obj is not None:
                            print(f"  Loaded from {start} to {stop_pos}")
                            return obj
                    except Exception:
                        continue
    except Exception as e:
        errors.append(f"STOP scan: {e}")

    raise RuntimeError(f"All load methods failed:\n" + "\n".join(f"  - {e}" for e in errors))


def try_load_with_unpickler(filepath):
    """Try loading with a custom unpickler that handles the two-object file format."""
    with open(filepath, 'rb') as f:
        raw = f.read()

    # The gen_data_code shows data_capture_decorator dumps a payload dict.
    # But the decorator also wraps the result if callable, and the decorator
    # itself might get pickled. Let's try to find the dict payload.

    # Strategy: find all valid pickle streams in the file
    results = []

    # Try loading from different offsets where pickle protocol headers appear
    i = 0
    while i < len(raw):
        if raw[i] == 0x80 and i + 1 < len(raw):
            proto = raw[i + 1]
            if proto <= 5:
                try:
                    buf = io.BytesIO(raw[i:])
                    obj = pickle.load(buf)
                    consumed = buf.tell()
                    results.append((i, i + consumed, obj))
                    i = i + consumed
                    continue
                except Exception:
                    pass
                try:
                    buf = io.BytesIO(raw[i:])
                    obj = dill.load(buf)
                    consumed = buf.tell()
                    results.append((i, i + consumed, obj))
                    i = i + consumed
                    continue
                except Exception:
                    pass
        i += 1

    return results


def deep_inspect(data, prefix="", max_depth=3):
    """Recursively inspect data structure."""
    if max_depth <= 0:
        print(f"{prefix}... (max depth)")
        return

    if isinstance(data, dict):
        print(f"{prefix}dict with {len(data)} keys: {list(data.keys())[:20]}")
        for k in list(data.keys())[:10]:
            v = data[k]
            print(f"{prefix}  [{k}]:", end=" ")
            if isinstance(v, np.ndarray):
                print(f"ndarray shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"dict keys={list(v.keys())[:10]}")
            elif isinstance(v, (list, tuple)):
                print(f"{type(v).__name__} len={len(v)}")
                for i, item in enumerate(v[:5]):
                    deep_inspect(item, prefix + "    ", max_depth - 1)
            elif hasattr(v, '__class__') and not isinstance(v, (int, float, str, bool, type(None))):
                cls = v.__class__
                print(f"{cls.__module__}.{cls.__name__}")
                if hasattr(v, '__dict__'):
                    for attr_k, attr_v in list(v.__dict__.items())[:5]:
                        if isinstance(attr_v, np.ndarray):
                            print(f"{prefix}    .{attr_k}: ndarray shape={attr_v.shape}")
                        else:
                            print(f"{prefix}    .{attr_k}: {type(attr_v).__name__}")
            else:
                print(f"{type(v).__name__} = {str(v)[:200]}")
    elif isinstance(data, (list, tuple)):
        print(f"{prefix}{type(data).__name__} len={len(data)}")
        for i, item in enumerate(data[:5]):
            print(f"{prefix}  [{i}]:", end=" ")
            deep_inspect(item, prefix + "    ", max_depth - 1)
    elif isinstance(data, np.ndarray):
        print(f"{prefix}ndarray shape={data.shape} dtype={data.dtype}")
    elif hasattr(data, '__class__') and not isinstance(data, (int, float, str, bool, type(None))):
        cls = data.__class__
        print(f"{prefix}{cls.__module__}.{cls.__name__}")
        if hasattr(data, '__dict__'):
            for k, v in list(data.__dict__.items())[:10]:
                print(f"{prefix}  .{k}:", end=" ")
                if isinstance(v, np.ndarray):
                    print(f"ndarray shape={v.shape}")
                elif isinstance(v, dict):
                    print(f"dict keys={list(v.keys())[:10]}")
                else:
                    print(f"{type(v).__name__} = {str(v)[:100]}")
    else:
        print(f"{prefix}{type(data).__name__} = {str(data)[:200]}")


def robust_load(filepath):
    """
    Most robust loading strategy - handles the case where dill.dump wrote
    None first (from decorator returning None) and then the actual payload.
    Also handles cases where the file was written by data_capture_decorator.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    print(f"  Raw file size: {len(raw)} bytes")
    print(f"  First 20 bytes: {raw[:20].hex()}")

    # The gen_data_code shows the decorator pattern:
    # 1. func is called, result is captured
    # 2. payload = {'func_name': ..., 'args': ..., 'kwargs': ..., 'output': ...}
    # 3. dill.dump(payload, f)
    #
    # But the decorator wraps functions with _data_capture_decorator_ and
    # _record_io_decorator_, which might interfere.
    #
    # The file has a specific format. Let's try to load it properly.

    # First, let's see how many pickle objects are in this file
    buf = io.BytesIO(raw)
    objects = []
    positions = []
    while buf.tell() < len(raw):
        pos = buf.tell()
        try:
            obj = dill.load(buf)
            objects.append(obj)
            positions.append(pos)
            end_pos = buf.tell()
            print(f"  Object at offset {pos}-{end_pos}: type={type(obj).__name__}", end="")
            if isinstance(obj, dict):
                print(f" keys={list(obj.keys())[:10]}", end="")
            if obj is None:
                print(" (None)", end="")
            print()
        except EOFError:
            break
        except Exception as e:
            # Try to skip ahead
            remaining = len(raw) - buf.tell()
            print(f"  Error at offset {pos}: {e}, remaining={remaining}")
            # Try to find next pickle header
            current = buf.tell()
            found = False
            for skip in range(1, min(remaining, 1000)):
                next_pos = current + skip
                if next_pos + 1 < len(raw) and raw[next_pos] == 0x80:
                    buf.seek(next_pos)
                    found = True
                    break
            if not found:
                break

    print(f"  Total objects found: {len(objects)}")

    # Look for the payload dict
    for i, obj in enumerate(objects):
        if isinstance(obj, dict) and 'args' in obj and 'output' in obj:
            print(f"  Using object {i} as payload")
            return obj

    # If we only got one object and it's None, the file might be truncated
    # or the data was appended after None
    if len(objects) == 1 and objects[0] is None:
        # The padding approach in the original code might have loaded None
        # because the file starts with a pickled None.
        # Let's try to load from after the first None
        end_of_first = positions[0] if len(positions) > 1 else 0
        buf = io.BytesIO(raw)
        try:
            dill.load(buf)  # skip first object
            first_end = buf.tell()
            remaining_raw = raw[first_end:]
            if len(remaining_raw) > 0:
                print(f"  Trying to load from offset {first_end}, remaining={len(remaining_raw)}")
                obj2 = dill.loads(remaining_raw)
                if obj2 is not None:
                    print(f"  Got second object: type={type(obj2).__name__}")
                    if isinstance(obj2, dict):
                        print(f"    keys={list(obj2.keys())}")
                    return obj2
        except Exception as e:
            print(f"  Failed to load second object: {e}")

    # If we have multiple objects, return the best one
    for obj in objects:
        if obj is not None:
            return obj

    # Last resort: try dill.loads on the raw bytes
    try:
        return dill.loads(raw)
    except Exception:
        pass

    raise RuntimeError("Could not load any valid object from file")


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
        elif basename == 'data_recon_slice.pkl' or basename == 'standard_data_recon_slice.pkl':
            outer_path = p

    if outer_path is None:
        # Try any remaining path
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
        outer_data = robust_load(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nLoaded data type: {type(outer_data).__name__}")
    deep_inspect(outer_data, prefix="  ")

    # Extract payload
    if isinstance(outer_data, dict):
        # Fix bytes keys
        keys = list(outer_data.keys())
        if keys and isinstance(keys[0], bytes):
            outer_data = {(k.decode('utf-8') if isinstance(k, bytes) else k): v for k, v in outer_data.items()}

        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)

        if isinstance(outer_kwargs, dict):
            kkeys = list(outer_kwargs.keys())
            if kkeys and isinstance(kkeys[0], bytes):
                outer_kwargs = {(k.decode('utf-8') if isinstance(k, bytes) else k): v for k, v in outer_kwargs.items()}
    else:
        print(f"FAIL: Loaded data is not a dict, it's {type(outer_data)}")
        sys.exit(1)

    print(f"\nOuter data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")

    # Debug args
    for i, arg in enumerate(outer_args):
        if isinstance(arg, np.ndarray):
            print(f"  arg[{i}]: ndarray shape={arg.shape}, dtype={arg.dtype}")
        elif isinstance(arg, str):
            print(f"  arg[{i}]: str = '{arg}'")
        elif hasattr(arg, '__class__'):
            cls_name = arg.__class__.__name__
            print(f"  arg[{i}]: {cls_name}")
            if hasattr(arg, 'reconstruct'):
                print(f"    -> has 'reconstruct' method")
        else:
            print(f"  arg[{i}]: {type(arg).__name__}")

    for k, v in outer_kwargs.items():
        if isinstance(v, np.ndarray):
            print(f"  kwarg[{k}]: ndarray shape={v.shape}, dtype={v.dtype}")
        elif hasattr(v, '__class__'):
            print(f"  kwarg[{k}]: {v.__class__.__name__}")
        else:
            print(f"  kwarg[{k}]: {type(v).__name__} = {str(v)[:100]}")

    if len(outer_args) == 0 and len(outer_kwargs) == 0:
        print("FAIL: No args/kwargs found. Cannot call recon_slice.")
        sys.exit(1)

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

    # --- Phase 3: Determine scenario and verify ---
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
                inner_data = robust_load(inner_path)
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

            print(f"  Inner args: {len(inner_args)}, kwargs: {list(inner_kwargs.keys())}")

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
            if result.size <= 10:
                print(f"Result values: {result}")
            else:
                print(f"Result sample: {result.flat[:5]}")
        if isinstance(expected, np.ndarray):
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            if expected.size <= 10:
                print(f"Expected values: {expected}")
            else:
                print(f"Expected sample: {expected.flat[:5]}")

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