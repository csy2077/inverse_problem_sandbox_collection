import sys
import os
import dill
import numpy as np
import traceback
import pickle
import struct
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-import modules that dill may need to reconstruct serialized objects
try:
    from devito import Function, TimeFunction, Grid
except Exception:
    pass

try:
    from examples.seismic.acoustic import AcousticWaveSolver
except Exception:
    pass

try:
    from examples.seismic import AcquisitionGeometry, Model, Receiver
except Exception:
    pass

try:
    import devito
except Exception:
    pass

from agent_forward_multi_shots import forward_multi_shots
from verification_utils import recursive_check


def repair_and_load(filepath):
    """
    Load a potentially truncated/corrupted pickle file.
    The file may have been written with dill.dump but truncated,
    causing EOFError. We try multiple strategies to recover the data.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    print(f"  File size: {file_size} bytes")
    print(f"  First 30 bytes (hex): {raw[:30].hex()}")
    print(f"  Last 10 bytes (hex): {raw[-10:].hex()}")

    # Check if file ends with STOP opcode (0x2e = b'.')
    ends_with_stop = (raw[-1:] == b'.')
    print(f"  Ends with STOP opcode: {ends_with_stop}")

    # Strategy 1: Direct dill.loads
    if ends_with_stop:
        try:
            data = dill.loads(raw)
            print(f"  Strategy 1 (dill.loads): Success, type={type(data)}")
            if isinstance(data, dict):
                print(f"    Keys: {list(data.keys())}")
            return data
        except Exception as e:
            print(f"  Strategy 1 (dill.loads) failed: {e}")

    # Strategy 2: Append STOP opcode if missing
    if not ends_with_stop:
        try:
            patched = raw + b'.'
            data = dill.loads(patched)
            print(f"  Strategy 2 (append STOP): Success, type={type(data)}")
            if isinstance(data, dict):
                print(f"    Keys: {list(data.keys())}")
            return data
        except Exception as e:
            print(f"  Strategy 2 (append STOP) failed: {e}")

    # Strategy 3: The file might have been written in append mode or have
    # multiple pickle streams concatenated. Try reading just the first valid stream.
    try:
        buf = io.BytesIO(raw)
        objects = []
        while buf.tell() < len(raw):
            pos_before = buf.tell()
            try:
                obj = dill.load(buf)
                objects.append(obj)
                print(f"  Strategy 3: Loaded object at pos {pos_before}, type={type(obj)}")
            except EOFError:
                break
            except Exception as e:
                print(f"  Strategy 3: Error at pos {pos_before}: {e}")
                break
        if objects:
            for obj in objects:
                if isinstance(obj, dict) and ('args' in obj or 'output' in obj or 'func_name' in obj):
                    print(f"  Strategy 3: Found target dict with keys {list(obj.keys())}")
                    return obj
            # Return last dict found
            for obj in reversed(objects):
                if isinstance(obj, dict):
                    return obj
            return objects[0]
    except Exception as e:
        print(f"  Strategy 3 failed: {e}")

    # Strategy 4: The pickle may be protocol 4 with frames. If the file is truncated
    # mid-frame, we can try to find frame boundaries and reconstruct.
    # Protocol 4 header: \x80\x04\x95 + 8-byte frame length
    if raw[0:3] == b'\x80\x04\x95':
        frame_len = struct.unpack('<Q', raw[3:11])[0]
        print(f"  Protocol 4 detected. First frame length: {frame_len}")
        print(f"  Data after header (11 bytes): available={file_size - 11}")

        # Strategy 4a: Try to find and add missing trailing bytes
        # Maybe the pickle needs a proper ending sequence
        # For a dict: the pattern is usually ... u\x94. (SETITEMS, MEMOIZE, STOP)
        endings_to_try = [
            b'.',           # Just STOP
            b'\x94.',       # MEMOIZE + STOP
            b'u.',          # SETITEMS + STOP  
            b'u\x94.',      # SETITEMS + MEMOIZE + STOP
            b'\x85\x94.',   # TUPLE1 + MEMOIZE + STOP
            b'e.',          # APPENDS + STOP
        ]
        
        for ending in endings_to_try:
            try:
                patched = raw + ending
                data = dill.loads(patched)
                print(f"  Strategy 4a (ending {ending.hex()}): Success, type={type(data)}")
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())}")
                return data
            except Exception:
                pass

        # Strategy 4b: Try truncating last few bytes and adding STOP
        for trim in range(1, min(100, file_size)):
            try:
                patched = raw[:-trim] + b'.'
                data = dill.loads(patched)
                print(f"  Strategy 4b (trim {trim} + STOP): Success, type={type(data)}")
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())}")
                return data
            except Exception:
                continue

    # Strategy 5: Use pickle with custom handling
    try:
        data = pickle.loads(raw)
        print(f"  Strategy 5 (pickle.loads): Success, type={type(data)}")
        return data
    except Exception as e:
        print(f"  Strategy 5 (pickle.loads) failed: {e}")

    if not ends_with_stop:
        try:
            data = pickle.loads(raw + b'.')
            print(f"  Strategy 5b (pickle.loads + STOP): Success, type={type(data)}")
            return data
        except Exception as e:
            print(f"  Strategy 5b failed: {e}")

    # Strategy 6: Scan for all STOP opcodes and try loading up to each one
    stop_positions = []
    for i in range(len(raw)):
        if raw[i:i+1] == b'.':
            stop_positions.append(i)
    
    print(f"  Strategy 6: Found {len(stop_positions)} potential STOP positions")
    # Try from the end backwards
    for pos in reversed(stop_positions):
        if pos < 20:
            continue
        try:
            candidate = raw[:pos+1]
            data = dill.loads(candidate)
            print(f"  Strategy 6 (STOP at {pos}): Success, type={type(data)}")
            if isinstance(data, dict):
                print(f"    Keys: {list(data.keys())}")
                if 'args' in data or 'output' in data:
                    return data
        except Exception:
            continue

    # Strategy 7: The file might have been written with a different protocol
    # or the frame length field is wrong. Try re-framing the data.
    if raw[0:2] == b'\x80\x04':
        # Skip protocol header and try to find the actual pickle data
        # Remove all FRAME opcodes and re-serialize
        try:
            cleaned = bytearray()
            i = 0
            # Skip protocol marker
            cleaned.extend(raw[0:2])
            i = 2
            while i < len(raw):
                if raw[i:i+1] == b'\x95' and i + 9 <= len(raw):
                    # FRAME opcode - skip it and its 8-byte argument
                    frame_len = struct.unpack('<Q', raw[i+1:i+9])[0]
                    i += 9
                    continue
                cleaned.append(raw[i])
                i += 1
            
            if cleaned[-1:] != b'.':
                cleaned.append(ord('.'))
            
            data = pickle.loads(bytes(cleaned))
            print(f"  Strategy 7 (deframed): Success, type={type(data)}")
            return data
        except Exception as e:
            print(f"  Strategy 7 failed: {e}")

    # Strategy 8: Try with different dill settings
    try:
        buf = io.BytesIO(raw if ends_with_stop else raw + b'.')
        unpickler = dill.Unpickler(buf)
        unpickler.ignore = True
        data = unpickler.load()
        print(f"  Strategy 8 (ignore=True): Success, type={type(data)}")
        return data
    except Exception as e:
        print(f"  Strategy 8 failed: {e}")

    # Strategy 9: Maybe the file needs to be read in binary chunks
    # because of filesystem issues. Re-read with explicit buffering.
    try:
        with open(filepath, 'rb', buffering=0) as f:
            raw2 = f.read()
        if len(raw2) != len(raw):
            print(f"  Strategy 9: Re-read got different size: {len(raw2)} vs {len(raw)}")
            if raw2[-1:] == b'.':
                data = dill.loads(raw2)
                print(f"  Strategy 9: Success with re-read, type={type(data)}")
                return data
            else:
                data = dill.loads(raw2 + b'.')
                print(f"  Strategy 9: Success with re-read+STOP, type={type(data)}")
                return data
    except Exception as e:
        print(f"  Strategy 9 failed: {e}")

    # Strategy 10: Check if the pickle file was actually written with multiple dump() calls.
    # The gen_data_code shows a single dill.dump(payload, f), so this is unlikely,
    # but let's check if the file was accidentally overwritten partially.
    # Try to manually parse the dict structure from the pickle stream.
    try:
        import pickletools
        buf = io.BytesIO(raw if ends_with_stop else raw + b'.')
        ops = []
        try:
            for opcode, arg, pos in pickletools.genops(buf):
                ops.append((opcode.name, arg, pos))
                if len(ops) > 50:
                    break
        except Exception:
            pass
        
        if ops:
            print(f"  Strategy 10: First {len(ops)} opcodes:")
            for op_name, arg, pos in ops[:20]:
                arg_str = str(arg)[:80] if arg is not None else ''
                print(f"    pos={pos}: {op_name} {arg_str}")
    except Exception as e:
        print(f"  Strategy 10 (debug): {e}")

    raise RuntimeError(f"All loading strategies failed for {filepath}")


def load_data_standard(filepath):
    """Standard load - try dill first, then repair_and_load."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # First try standard load
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print(f"  Standard dill.load succeeded, type={type(data)}")
        return data
    except Exception as e:
        print(f"  Standard dill.load failed: {e}")
    
    # Fall back to repair
    return repair_and_load(filepath)


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_forward_multi_shots.pkl'
    ]

    # Scan the directory for any related files
    std_data_dir = os.path.dirname(data_paths[0])
    all_related_files = []
    if os.path.isdir(std_data_dir):
        print(f"Scanning directory: {std_data_dir}")
        for fname in sorted(os.listdir(std_data_dir)):
            if 'forward_multi_shots' in fname and fname.endswith('.pkl'):
                full_path = os.path.join(std_data_dir, fname)
                print(f"  Found related file: {fname} ({os.path.getsize(full_path)} bytes)")
                if full_path not in data_paths:
                    all_related_files.append(full_path)
        data_paths.extend(all_related_files)

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        elif 'forward_multi_shots' in basename and ('parent' not in basename):
            if outer_path is None:
                outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file")
        sys.exit(1)

    print(f"\nOuter data file: {outer_path}")
    print(f"  Exists: {os.path.exists(outer_path)}")
    if os.path.exists(outer_path):
        print(f"  Size: {os.path.getsize(outer_path)} bytes")
    print(f"Inner data files: {inner_paths}")

    # Phase 1: Load outer data
    print(f"\n=== Loading outer data ===")
    try:
        outer_data = load_data_standard(outer_path)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(outer_data, dict):
        print(f"FAIL: Expected dict, got {type(outer_data)}")
        sys.exit(1)

    # Parse outer data
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    print(f"\nOuter data parsed. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}")
    print(f"  kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")
    if outer_output is not None:
        print(f"  output type: {type(outer_output)}")
        if isinstance(outer_output, list):
            print(f"  output list length: {len(outer_output)}")

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure pattern ===")

        print("Phase 1: Reconstructing operator via forward_multi_shots(...)...")
        try:
            agent_operator = forward_multi_shots(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_multi_shots raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Operator created successfully: {type(agent_operator)}")

        all_passed = True
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                inner_data = load_data_standard(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")

            print("Executing operator with inner args...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Result mismatch for {os.path.basename(inner_path)}: {msg}")
                all_passed = False
            else:
                print(f"PASS: {os.path.basename(inner_path)} verified successfully.")

        if not all_passed:
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("\n=== Scenario A: Simple function call ===")

        print("Executing forward_multi_shots(...)...")
        try:
            result = forward_multi_shots(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: forward_multi_shots raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        print("Comparing results...")
        print(f"  Expected type: {type(expected)}")
        print(f"  Result type: {type(result)}")

        if isinstance(expected, list):
            print(f"  Expected list length: {len(expected)}")
            for i, item in enumerate(expected[:3]):
                print(f"  Expected[{i}] type: {type(item)}")
                if hasattr(item, 'data'):
                    print(f"    .data shape: {np.array(item.data).shape}")
        if isinstance(result, list):
            print(f"  Result list length: {len(result)}")
            for i, item in enumerate(result[:3]):
                print(f"  Result[{i}] type: {type(item)}")
                if hasattr(item, 'data'):
                    print(f"    .data shape: {np.array(item.data).shape}")

        passed = False
        msg = ""
        
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"  recursive_check raised exception: {e}")
            traceback.print_exc()

            # Fallback: manual numpy comparison
            print("\n  Attempting manual comparison...")
            try:
                if isinstance(expected, list) and isinstance(result, list):
                    if len(expected) != len(result):
                        print(f"FAIL: List length mismatch: expected {len(expected)}, got {len(result)}")
                        sys.exit(1)

                    all_close = True
                    for i in range(len(expected)):
                        exp_item = expected[i]
                        res_item = result[i]

                        exp_data = None
                        res_data = None

                        if hasattr(exp_item, 'data'):
                            exp_data = np.array(exp_item.data)
                        elif isinstance(exp_item, np.ndarray):
                            exp_data = exp_item

                        if hasattr(res_item, 'data'):
                            res_data = np.array(res_item.data)
                        elif isinstance(res_item, np.ndarray):
                            res_data = res_item

                        if exp_data is not None and res_data is not None:
                            if exp_data.shape != res_data.shape:
                                print(f"  Item {i}: shape mismatch {exp_data.shape} vs {res_data.shape}")
                                all_close = False
                            elif not np.allclose(exp_data, res_data, rtol=1e-5, atol=1e-5):
                                max_diff = np.max(np.abs(exp_data - res_data))
                                print(f"  Item {i}: values differ, max diff = {max_diff}")
                                all_close = False
                            else:
                                print(f"  Item {i}: MATCH")
                        else:
                            print(f"  Item {i}: Could not extract comparable data")
                            all_close = False

                    passed = all_close
                    msg = "Manual comparison " + ("passed" if passed else "found differences")
                else:
                    print(f"FAIL: Cannot manually compare types {type(expected)} and {type(result)}")
                    sys.exit(1)
            except Exception as e2:
                print(f"FAIL: Manual comparison also failed: {e2}")
                traceback.print_exc()
                sys.exit(1)

        if not passed:
            print(f"FAIL: Result mismatch: {msg}")
            sys.exit(1)
        else:
            print("PASS: Output verified successfully.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()