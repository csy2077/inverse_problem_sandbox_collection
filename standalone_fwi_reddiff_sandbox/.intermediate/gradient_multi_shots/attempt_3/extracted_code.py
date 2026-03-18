import sys
import os
import dill
import numpy as np
import traceback
import struct

# Ensure the working directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to make the fg_pair class available for dill deserialization
class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        return fg_pair(f, g)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

# Inject fg_pair into multiple modules so dill can resolve it
import builtins
builtins.fg_pair = fg_pair

try:
    import __main__
    __main__.fg_pair = fg_pair
except Exception:
    pass

try:
    import agent_gradient_multi_shots
    agent_gradient_multi_shots.fg_pair = fg_pair
except Exception:
    pass

from agent_gradient_multi_shots import gradient_multi_shots
from verification_utils import recursive_check


def load_data(filepath):
    """Load data file using dill, handling potential truncation or multiple dumps."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")

    print(f"  Loading: {filepath} ({file_size} bytes)")

    # Read entire file content
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    print(f"  Raw data length: {len(raw_data)} bytes")
    print(f"  First 20 bytes hex: {raw_data[:20].hex()}")

    errors = []

    # Method 1: Standard dill.load from beginning
    try:
        import io
        buf = io.BytesIO(raw_data)
        data = dill.load(buf)
        print(f"  dill.load succeeded at position {buf.tell()} of {len(raw_data)}")
        return data
    except Exception as e:
        errors.append(f"dill.load: {e}")
        print(f"  dill.load failed: {e}")

    # Method 2: Try loading with recurse=True
    try:
        import io
        dill.settings['recurse'] = True
        buf = io.BytesIO(raw_data)
        data = dill.load(buf)
        print(f"  dill.load(recurse) succeeded")
        return data
    except Exception as e:
        errors.append(f"dill.load(recurse): {e}")
        print(f"  dill.load(recurse) failed: {e}")
    finally:
        dill.settings['recurse'] = False

    # Method 3: pickle
    try:
        import pickle
        import io
        buf = io.BytesIO(raw_data)
        data = pickle.load(buf)
        return data
    except Exception as e:
        errors.append(f"pickle.load: {e}")
        print(f"  pickle.load failed: {e}")

    # Method 4: Scan for pickle opcodes - the file might have a prefix/header
    # Look for pickle protocol markers
    try:
        import io
        # Search for pickle start markers (protocol 2: \x80\x02, protocol 4: \x80\x04, protocol 5: \x80\x05)
        for protocol in [5, 4, 3, 2]:
            marker = bytes([0x80, protocol])
            idx = 0
            while idx < len(raw_data):
                pos = raw_data.find(marker, idx)
                if pos == -1:
                    break
                if pos > 0:
                    try:
                        buf = io.BytesIO(raw_data[pos:])
                        data = dill.load(buf)
                        print(f"  Loaded from offset {pos} with protocol {protocol}")
                        return data
                    except Exception:
                        pass
                idx = pos + 1
    except Exception as e:
        errors.append(f"offset scan: {e}")
        print(f"  Offset scan failed: {e}")

    # Method 5: Try loading multiple objects (file may contain multiple pickled objects)
    try:
        import io
        buf = io.BytesIO(raw_data)
        objects = []
        while buf.tell() < len(raw_data):
            try:
                obj = dill.load(buf)
                objects.append(obj)
            except EOFError:
                break
            except Exception:
                break
        if objects:
            if len(objects) == 1:
                return objects[0]
            # If multiple objects, try to reconstruct as dict
            print(f"  Loaded {len(objects)} objects")
            return objects
    except Exception as e:
        errors.append(f"multi-load: {e}")

    # Method 6: Try with gzip decompression
    try:
        import gzip
        import io
        decompressed = gzip.decompress(raw_data)
        buf = io.BytesIO(decompressed)
        data = dill.load(buf)
        print(f"  Loaded after gzip decompression")
        return data
    except Exception as e:
        errors.append(f"gzip+dill: {e}")

    # Method 7: Try with zlib decompression
    try:
        import zlib
        import io
        decompressed = zlib.decompress(raw_data)
        buf = io.BytesIO(decompressed)
        data = dill.load(buf)
        print(f"  Loaded after zlib decompression")
        return data
    except Exception as e:
        errors.append(f"zlib+dill: {e}")

    # Method 8: The file might be the "standard_data" variant
    # Check if there's a standard_data version
    data_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    alt_name = "standard_" + basename
    alt_path = os.path.join(data_dir, alt_name)
    if os.path.exists(alt_path) and alt_path != filepath:
        try:
            with open(alt_path, 'rb') as f:
                data = dill.load(f)
            print(f"  Loaded from alternative path: {alt_path}")
            return data
        except Exception as e:
            errors.append(f"alt_path: {e}")

    # Method 9: Try reading as numpy file
    try:
        import io
        buf = io.BytesIO(raw_data)
        data = np.load(buf, allow_pickle=True)
        return data
    except Exception as e:
        errors.append(f"np.load: {e}")

    raise RuntimeError(f"All load methods failed for {filepath}: {'; '.join(errors)}")


def try_load_from_directory(data_dir, target_name):
    """Try to find and load any matching pkl file in the directory."""
    if not os.path.isdir(data_dir):
        return None

    candidates = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.pkl') and 'gradient_multi_shots' in f:
            fpath = os.path.join(data_dir, f)
            fsize = os.path.getsize(fpath)
            if fsize > 0:
                candidates.append((fpath, fsize))
                print(f"  Found candidate: {f} ({fsize} bytes)")

    for fpath, fsize in candidates:
        try:
            data = load_data(fpath)
            if isinstance(data, dict) and ('args' in data or 'output' in data):
                print(f"  Successfully loaded: {os.path.basename(fpath)}")
                return data, fpath
        except Exception as e:
            print(f"  Failed to load {os.path.basename(fpath)}: {e}")
            continue

    return None


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data/data_gradient_multi_shots.pkl'
    ]

    outer_path = data_paths[0]
    data_dir = os.path.dirname(outer_path)

    # Check for standard_data variant
    standard_path = os.path.join(data_dir, 'standard_data_gradient_multi_shots.pkl')

    # Scan for all relevant files
    print("Scanning data directory for relevant files...")
    all_relevant = []
    inner_paths = []
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.pkl') and 'gradient_multi_shots' in f:
                fpath = os.path.join(data_dir, f)
                fsize = os.path.getsize(fpath)
                all_relevant.append((f, fsize))
                print(f"  Found: {f} ({fsize} bytes)")
                if 'parent_function' in f or 'parent_' in f:
                    if fsize > 0:
                        inner_paths.append(fpath)

    print(f"\nOuter path: {outer_path}")
    print(f"Standard path exists: {os.path.exists(standard_path)}")
    print(f"Inner paths: {[os.path.basename(p) for p in inner_paths]}")

    # Try multiple paths to load data
    outer_data = None
    load_paths_to_try = [outer_path]
    if os.path.exists(standard_path) and standard_path != outer_path:
        load_paths_to_try.insert(0, standard_path)

    # Also check run_code directory for the pkl
    run_code_dir = os.path.dirname(data_dir)
    alt_data_path = os.path.join(run_code_dir, 'data_gradient_multi_shots.pkl')
    if os.path.exists(alt_data_path):
        load_paths_to_try.append(alt_data_path)

    for path in load_paths_to_try:
        if not os.path.exists(path):
            continue
        try:
            outer_data = load_data(path)
            print(f"Successfully loaded from: {path}")
            break
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

    if outer_data is None:
        # Last resort: try to find any loadable file
        print("\nTrying all candidates in directory...")
        result = try_load_from_directory(data_dir, 'gradient_multi_shots')
        if result is not None:
            outer_data, loaded_path = result
            print(f"Loaded from fallback: {loaded_path}")

    if outer_data is None:
        # The file exists but can't be loaded - it might be corrupted/truncated
        # Let's check if maybe it was saved with a different serializer or needs special handling
        print(f"\nFAIL: Could not load any data file. Checking file integrity...")

        # Print file details for debugging
        if os.path.exists(outer_path):
            with open(outer_path, 'rb') as f:
                header = f.read(100)
                print(f"  File header (hex): {header[:50].hex()}")
                print(f"  File header (repr): {repr(header[:50])}")

                # Check if it looks like it might be a partial write
                f.seek(0, 2)
                total = f.tell()
                f.seek(max(0, total - 50))
                tail = f.read()
                print(f"  File tail (hex): {tail.hex()}")

                # Try to detect if file is truncated by checking for pickle STOP opcode
                f.seek(0)
                content = f.read()
                # pickle STOP opcode is '.'  (0x2e)
                stop_positions = [i for i, b in enumerate(content) if b == 0x2e]
                if stop_positions:
                    last_stop = stop_positions[-1]
                    print(f"  Last STOP opcode at position {last_stop} of {total}")
                    if last_stop < total - 1:
                        print(f"  File has {total - last_stop - 1} bytes after last STOP")
                        # Try loading up to the STOP opcode
                        try:
                            import io
                            buf = io.BytesIO(content[:last_stop + 1])
                            outer_data = dill.load(buf)
                            print(f"  Successfully loaded truncated file up to STOP opcode")
                        except Exception as e:
                            print(f"  Truncated load failed: {e}")

    if outer_data is None:
        print("FAIL: Could not load outer data from any source")
        sys.exit(1)

    if not isinstance(outer_data, dict):
        # Maybe it's a list or other structure
        print(f"Data type: {type(outer_data)}")
        if isinstance(outer_data, (list, tuple)):
            if len(outer_data) > 0 and isinstance(outer_data[0], dict):
                outer_data = outer_data[0]
            else:
                print(f"FAIL: Unexpected data structure: {type(outer_data)}")
                sys.exit(1)
        else:
            print(f"FAIL: Unexpected outer data type: {type(outer_data)}")
            sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"\nOuter func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if inner_paths:
        # --- Scenario B: Factory/Closure pattern ---
        print("\nScenario B: Factory/Closure pattern detected.")

        try:
            print("Running gradient_multi_shots to get operator...")
            agent_operator = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Operator type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"\nProcessing inner: {os.path.basename(inner_path)}")
            try:
                inner_data = load_data(inner_path)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            print(f"PASSED for {os.path.basename(inner_path)}")

    else:
        # --- Scenario A: Simple function call ---
        print("\nScenario A: Simple function call.")

        try:
            print("Running gradient_multi_shots...")
            result = gradient_multi_shots(*outer_args, **outer_kwargs)
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: gradient_multi_shots raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()