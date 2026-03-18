import sys
import os
import dill
import numpy as np
import traceback
import pickle
import io
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
    """Load a potentially truncated pickle file by reading multiple concatenated objects."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")

    print(f"  Loading: {filepath} ({file_size} bytes)")

    with open(filepath, 'rb') as f:
        raw_data = f.read()

    print(f"  Raw data length: {len(raw_data)} bytes")
    print(f"  First 20 bytes hex: {raw_data[:20].hex()}")

    # The file header shows pickle protocol 4 (\x80\x04)
    # The frame length bytes follow: \x95 then 8-byte little-endian length
    # The issue is that the pickle frame header declares a length larger than the actual data,
    # OR the file contains multiple pickled objects concatenated.

    # Strategy: The pickle STOP opcode is at position 8993405, with 1151 trailing bytes.
    # This suggests the file might have the main dict pickled with a frame that references
    # large objects that were pickled separately or the frame length is wrong.

    # Let's try approach: find all pickle start positions and load objects individually,
    # then reconstruct the dict.

    # First, let's try loading with protocol awareness - fix the frame length
    # Protocol 4 format: \x80\x04\x95<8-byte-le-frame-length>
    # The frame length might be wrong (declaring more data than exists)

    # Approach 1: Fix the frame length in the header
    if len(raw_data) >= 11 and raw_data[0:2] == b'\x80\x04' and raw_data[2] == 0x95:
        declared_frame_len = struct.unpack('<Q', raw_data[3:11])[0]
        actual_remaining = len(raw_data) - 11
        print(f"  Pickle protocol 4, declared frame length: {declared_frame_len}, actual remaining: {actual_remaining}")

        if declared_frame_len > actual_remaining:
            # Frame length is larger than available data - file is truncated or frame length is wrong
            # Try fixing the frame length
            fixed_data = raw_data[:3] + struct.pack('<Q', actual_remaining) + raw_data[11:]
            try:
                buf = io.BytesIO(fixed_data)
                data = pickle.load(buf)
                print(f"  Loaded with fixed frame length")
                return data
            except Exception as e:
                print(f"  Fixed frame length load failed: {e}")

            # Try with the data up to the STOP opcode
            # Find the last STOP opcode (0x2e = '.')
            last_stop = raw_data.rfind(b'.')
            if last_stop > 0:
                truncated = raw_data[:last_stop + 1]
                # Fix frame length for truncated data
                trunc_remaining = len(truncated) - 11
                fixed_trunc = truncated[:3] + struct.pack('<Q', trunc_remaining) + truncated[11:]
                try:
                    buf = io.BytesIO(fixed_trunc)
                    data = pickle.load(buf)
                    print(f"  Loaded truncated+fixed data (up to STOP at {last_stop})")
                    return data
                except Exception as e:
                    print(f"  Truncated+fixed load failed: {e}")

    # Approach 2: The file may contain multiple pickle streams concatenated.
    # The first stream might be the metadata dict (small), and subsequent streams are large objects.
    # Let's find all pickle protocol markers and try loading each.
    pickle_starts = []
    for i in range(len(raw_data) - 2):
        if raw_data[i] == 0x80 and raw_data[i + 1] in (2, 3, 4, 5):
            pickle_starts.append(i)

    print(f"  Found {len(pickle_starts)} potential pickle start positions")
    if len(pickle_starts) > 1:
        print(f"  First few start positions: {pickle_starts[:10]}")

    # Try loading from each start position
    loaded_objects = []
    for start_pos in pickle_starts:
        try:
            buf = io.BytesIO(raw_data[start_pos:])
            obj = pickle.load(buf)
            end_pos = start_pos + buf.tell()
            loaded_objects.append((start_pos, end_pos, obj))
            print(f"  Loaded object at offset {start_pos}-{end_pos}, type: {type(obj).__name__}")
        except Exception:
            pass

    if loaded_objects:
        # Check if any single object is the complete dict we need
        for start_pos, end_pos, obj in loaded_objects:
            if isinstance(obj, dict) and 'func_name' in obj and 'args' in obj and 'output' in obj:
                print(f"  Found complete data dict at offset {start_pos}")
                return obj

        # Check if any is a dict with at least func_name
        for start_pos, end_pos, obj in loaded_objects:
            if isinstance(obj, dict) and 'func_name' in obj:
                print(f"  Found partial data dict at offset {start_pos}, keys: {list(obj.keys())}")
                return obj

        # If the first object is a dict, it might be incomplete - try to fill in from others
        if isinstance(loaded_objects[0][2], dict):
            result_dict = loaded_objects[0][2]
            print(f"  First dict keys: {list(result_dict.keys())}")
            return result_dict

    # Approach 3: Try dill with different settings
    for byref in [True, False]:
        try:
            dill.settings['byref'] = byref
            buf = io.BytesIO(raw_data)
            data = dill.load(buf)
            print(f"  dill.load(byref={byref}) succeeded")
            return data
        except Exception as e:
            print(f"  dill.load(byref={byref}) failed: {e}")
        finally:
            dill.settings['byref'] = False

    # Approach 4: Manually parse the pickle to extract the dict
    # The header shows: {func_name: "gradient_multi_shots", ...}
    # Let's try to use pickletools to understand the structure
    try:
        import pickletools
        buf = io.BytesIO(raw_data[:min(500, len(raw_data))])
        print("  Pickle disassembly (first 500 bytes):")
        pickletools.dis(buf, annotate=1)
    except Exception as e:
        print(f"  pickletools.dis failed: {e}")

    # Approach 5: The file has a STOP at 8993405. Try loading just that portion
    # but WITHOUT the frame opcode (skip the frame header)
    last_stop = raw_data.rfind(b'.')
    if last_stop > 0:
        # Try skipping the frame header entirely - start from offset 11 (after \x80\x04\x95 + 8 bytes)
        if len(raw_data) >= 11:
            # Reconstruct with correct frame length
            content_data = raw_data[11:last_stop + 1]
            new_header = b'\x80\x04\x95' + struct.pack('<Q', len(content_data))
            fixed = new_header + content_data
            try:
                buf = io.BytesIO(fixed)
                data = pickle.load(buf)
                print(f"  Loaded with reconstructed frame (content up to STOP)")
                return data
            except Exception as e:
                print(f"  Reconstructed frame load failed: {e}")

            # Try protocol 2 wrapper
            try:
                new_data = b'\x80\x02' + content_data
                buf = io.BytesIO(new_data)
                data = pickle.load(buf)
                print(f"  Loaded with protocol 2 wrapper")
                return data
            except Exception as e:
                print(f"  Protocol 2 wrapper failed: {e}")

    # Approach 6: Read the file in chunks to avoid memory mapping issues
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        print(f"  Direct file dill.load succeeded")
        return data
    except Exception as e:
        print(f"  Direct file dill.load failed: {e}")

    # Approach 7: mmap-based loading
    try:
        import mmap
        with open(filepath, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            buf = io.BytesIO(mm[:])
            data = dill.load(buf)
            mm.close()
            print(f"  mmap-based load succeeded")
            return data
    except Exception as e:
        print(f"  mmap-based load failed: {e}")

    raise RuntimeError(f"All load methods failed for {filepath}")


def load_data_robust(filepath):
    """
    Robust loader that handles truncated pickle files by loading individual
    pickled objects and reconstructing the data dict.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    file_size = os.path.getsize(filepath)
    print(f"  Attempting robust load of {filepath} ({file_size} bytes)")

    with open(filepath, 'rb') as f:
        raw_data = f.read()

    # The file starts with \x80\x04\x95 (pickle4 + FRAME opcode)
    # followed by 8-byte frame length, then the actual data
    # The problem: frame length says data is longer than what's in the file
    # This typically happens when dill serializes large objects and the
    # frame encompasses the entire stream

    # Key insight from the error output:
    # - declared frame length (from bytes 3-11) is larger than file
    # - STOP opcode at 8993405, file is 8994557
    # - After STOP there are 1151 zero bytes (padding or another object)

    # Let's try: remove the FRAME opcode entirely and let pickle handle it
    if len(raw_data) >= 11 and raw_data[0:2] == b'\x80\x04' and raw_data[2] == 0x95:
        declared_frame_len = struct.unpack('<Q', raw_data[3:11])[0]
        actual_data_len = len(raw_data) - 11
        print(f"  Frame length declared: {declared_frame_len}, actual: {actual_data_len}")

        # Strategy: Remove the FRAME opcode and keep protocol header + content
        # \x80\x04 (protocol) + content (skip \x95 + 8 byte length)
        no_frame_data = raw_data[0:2] + raw_data[11:]

        # Find the STOP opcode in the no-frame data
        last_stop = no_frame_data.rfind(b'.')
        if last_stop > 0:
            trimmed = no_frame_data[:last_stop + 1]
            try:
                buf = io.BytesIO(trimmed)
                data = pickle.load(buf)
                print(f"  No-frame trimmed load succeeded")
                return data
            except Exception as e:
                print(f"  No-frame trimmed load failed: {e}")

        try:
            buf = io.BytesIO(no_frame_data)
            data = pickle.load(buf)
            print(f"  No-frame load succeeded")
            return data
        except Exception as e:
            print(f"  No-frame load failed: {e}")

        # Try: just fix the frame length to match actual data
        # Trim trailing zeros first
        trimmed_raw = raw_data.rstrip(b'\x00')
        if len(trimmed_raw) < len(raw_data):
            print(f"  Trimmed {len(raw_data) - len(trimmed_raw)} trailing zero bytes")
            # Make sure it ends with STOP
            if not trimmed_raw.endswith(b'.'):
                last_stop_pos = trimmed_raw.rfind(b'.')
                if last_stop_pos > 0:
                    trimmed_raw = trimmed_raw[:last_stop_pos + 1]

            content_len = len(trimmed_raw) - 11
            fixed = trimmed_raw[:3] + struct.pack('<Q', content_len) + trimmed_raw[11:]
            try:
                buf = io.BytesIO(fixed)
                data = pickle.load(buf)
                print(f"  Trimmed+fixed load succeeded")
                return data
            except Exception as e:
                print(f"  Trimmed+fixed load failed: {e}")

    # Try the Unpickler approach with a custom file object that doesn't EOF
    class PaddedReader:
        """Wraps raw data and pads with zeros if read goes past end."""
        def __init__(self, data, pad_size=1024*1024):
            self.data = data
            self.pos = 0
            self.pad_size = pad_size
            self.total = len(data) + pad_size

        def read(self, n=-1):
            if n == -1:
                result = self.data[self.pos:]
                self.pos = len(self.data)
                return result
            end = self.pos + n
            if end <= len(self.data):
                result = self.data[self.pos:end]
            else:
                result = self.data[self.pos:] + b'\x00' * (end - len(self.data))
            self.pos = end
            return result

        def readline(self):
            idx = self.data.find(b'\n', self.pos)
            if idx == -1:
                result = self.data[self.pos:]
                self.pos = len(self.data)
            else:
                result = self.data[self.pos:idx + 1]
                self.pos = idx + 1
            return result

        def tell(self):
            return self.pos

        def seek(self, pos, whence=0):
            if whence == 0:
                self.pos = pos
            elif whence == 1:
                self.pos += pos
            elif whence == 2:
                self.pos = len(self.data) + pos

    # Try loading with padded reader
    try:
        reader = PaddedReader(raw_data, pad_size=declared_frame_len - actual_data_len + 1024)
        data = pickle.load(reader)
        print(f"  Padded reader load succeeded")
        return data
    except Exception as e:
        print(f"  Padded reader load failed: {e}")

    # Try with dill and padded reader
    try:
        reader = PaddedReader(raw_data, pad_size=declared_frame_len - actual_data_len + 1024)
        data = dill.load(reader)
        print(f"  Padded reader dill load succeeded")
        return data
    except Exception as e:
        print(f"  Padded reader dill load failed: {e}")

    raise RuntimeError(f"Robust load failed for {filepath}")


def try_manual_reconstruction(filepath):
    """
    Last resort: manually parse the pickle stream to extract what we can.
    The pickle file contains a dict with keys: func_name, args, kwargs, output.
    We know func_name = 'gradient_multi_shots'.
    """
    print(f"  Attempting manual reconstruction from {filepath}")

    with open(filepath, 'rb') as f:
        raw_data = f.read()

    # The data structure is a dict saved by _data_capture_decorator_
    # payload = {'func_name': func_name, 'args': args, 'kwargs': kwargs, 'output': result}

    # Try using an Unpickler that catches the truncation
    class TolerantUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle fg_pair
            if name == 'fg_pair':
                return fg_pair
            return super().find_class(module, name)

    # The key issue is the FRAME opcode declares a length that exceeds the file.
    # In Python's pickle, the FRAME opcode just pre-reads that many bytes for buffering.
    # If we patch the C implementation or use the Python implementation, we might succeed.

    # Force Python pickle implementation
    try:
        buf = io.BytesIO(raw_data)
        unpickler = pickle._Unpickler(buf)
        unpickler.find_class = lambda module, name: fg_pair if name == 'fg_pair' else pickle._Unpickler.find_class(unpickler, module, name)
        data = unpickler.load()
        print(f"  Python _Unpickler succeeded")
        return data
    except Exception as e:
        print(f"  Python _Unpickler failed: {e}")

    # Try with fixed frame length using Python unpickler
    if len(raw_data) >= 11 and raw_data[2] == 0x95:
        # Fix frame length
        actual_content = len(raw_data) - 11
        fixed = raw_data[:3] + struct.pack('<Q', actual_content) + raw_data[11:]
        try:
            buf = io.BytesIO(fixed)
            unpickler = pickle._Unpickler(buf)
            data = unpickler.load()
            print(f"  Fixed frame Python _Unpickler succeeded")
            return data
        except Exception as e:
            print(f"  Fixed frame Python _Unpickler failed: {e}")

        # Trim to last STOP and fix
        last_stop = raw_data.rfind(b'.')
        if last_stop > 10:
            trimmed = raw_data[:last_stop + 1]
            content_len = len(trimmed) - 11
            fixed = trimmed[:3] + struct.pack('<Q', content_len) + trimmed[11:]
            try:
                buf = io.BytesIO(fixed)
                unpickler = pickle._Unpickler(buf)
                data = unpickler.load()
                print(f"  Trimmed+fixed Python _Unpickler succeeded")
                return data
            except Exception as e:
                print(f"  Trimmed+fixed Python _Unpickler failed: {e}")

    raise RuntimeError("Manual reconstruction failed")


def load_with_extended_data(filepath):
    """
    The file's FRAME declares a length larger than actual file content.
    This means pickle expects more data. The data after STOP opcode (zeros)
    suggests the file was partially written - but the STOP opcode IS there,
    meaning the pickle stream itself is complete, just the FRAME length was
    computed incorrectly (perhaps based on estimated size before actual serialization).

    Solution: Set frame length to match the data up to and including the STOP opcode.
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) < 11:
        raise ValueError("File too small")

    # Verify protocol 4 with FRAME
    if raw_data[0:2] != b'\x80\x04' or raw_data[2] != 0x95:
        raise ValueError("Not a protocol 4 pickle with FRAME")

    declared_frame_len = struct.unpack('<Q', raw_data[3:11])[0]
    print(f"  Declared frame length: {declared_frame_len}")
    print(f"  File size minus header: {len(raw_data) - 11}")

    # Find the STOP opcode
    # The STOP opcode (b'.') marks the end of the pickle stream
    # Search backwards from end
    last_stop = -1
    for i in range(len(raw_data) - 1, 10, -1):
        if raw_data[i] == 0x2e:  # '.'
            last_stop = i
            break

    if last_stop < 0:
        raise ValueError("No STOP opcode found")

    print(f"  Last STOP opcode at position: {last_stop}")

    # The frame content is from position 11 to last_stop (inclusive)
    frame_content_len = last_stop - 11 + 1
    print(f"  Actual frame content length: {frame_content_len}")

    # Reconstruct with correct frame length
    fixed_data = raw_data[:3] + struct.pack('<Q', frame_content_len) + raw_data[11:last_stop + 1]
    print(f"  Fixed data size: {len(fixed_data)}")

    # Try pickle first (faster)
    try:
        buf = io.BytesIO(fixed_data)
        data = pickle.loads(fixed_data)
        print(f"  pickle.loads with fixed frame succeeded")
        return data
    except Exception as e:
        print(f"  pickle.loads with fixed frame failed: {e}")

    # Try dill
    try:
        data = dill.loads(fixed_data)
        print(f"  dill.loads with fixed frame succeeded")
        return data
    except Exception as e:
        print(f"  dill.loads with fixed frame failed: {e}")

    # Try without frame at all - just protocol header + content
    no_frame = raw_data[0:2] + raw_data[11:last_stop + 1]
    try:
        data = pickle.loads(no_frame)
        print(f"  pickle.loads without frame succeeded")
        return data
    except Exception as e:
        print(f"  pickle.loads without frame failed: {e}")

    try:
        data = dill.loads(no_frame)
        print(f"  dill.loads without frame succeeded")
        return data
    except Exception as e:
        print(f"  dill.loads without frame failed: {e}")

    # Maybe the issue is that there are MULTIPLE frames
    # Let's try to find and fix all FRAME opcodes
    frame_positions = []
    i = 0
    while i < len(raw_data) - 9:
        if raw_data[i] == 0x95:  # FRAME opcode
            frame_len = struct.unpack('<Q', raw_data[i+1:i+9])[0]
            frame_positions.append((i, frame_len))
            i += 9
        else:
            i += 1

    if len(frame_positions) > 1:
        print(f"  Found {len(frame_positions)} FRAME opcodes at positions: {[p[0] for p in frame_positions]}")

    # Try protocol 5
    proto5_data = b'\x80\x05\x95' + struct.pack('<Q', frame_content_len) + raw_data[11:last_stop + 1]
    try:
        data = pickle.loads(proto5_data)
        print(f"  protocol 5 load succeeded")
        return data
    except Exception as e:
        print(f"  protocol 5 load failed: {e}")

    raise RuntimeError("load_with_extended_data failed")


def monkey_patch_and_load(filepath):
    """
    Monkey-patch pickle's frame reading to handle truncated frames,
    then load the file.
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    # Find the actual STOP position
    last_stop = raw_data.rfind(b'.')
    if last_stop <= 0:
        raise ValueError("No STOP opcode found")

    # Take data up to STOP (inclusive)
    data_to_load = raw_data[:last_stop + 1]

    # Patch: Replace the FRAME length with the actual remaining data length
    if len(data_to_load) >= 11 and data_to_load[0:2] == b'\x80\x04' and data_to_load[2] == 0x95:
        content_after_frame_header = len(data_to_load) - 11
        patched = data_to_load[:3] + struct.pack('<Q', content_after_frame_header) + data_to_load[11:]

        # Use dill.loads which handles custom classes better
        try:
            data = dill.loads(patched)
            print(f"  monkey_patch dill.loads succeeded")
            return data
        except Exception as e:
            print(f"  monkey_patch dill.loads failed: {e}")
            traceback.print_exc()

        try:
            data = pickle.loads(patched)
            print(f"  monkey_patch pickle.loads succeeded")
            return data
        except Exception as e:
            print(f"  monkey_patch pickle.loads failed: {e}")
            traceback.print_exc()

    raise RuntimeError("monkey_patch_and_load failed")


def load_by_overriding_framing(filepath):
    """
    Override the internal framing mechanism by using _Unpickler
    with a BytesIO that has the correct content.
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    last_stop = raw_data.rfind(b'.')
    if last_stop <= 0:
        raise ValueError("No STOP opcode")

    # Remove all FRAME opcodes from the data
    # FRAME opcode is 0x95 followed by 8-byte length
    cleaned = bytearray()
    i = 0
    # Keep protocol header
    if raw_data[0] == 0x80:
        cleaned.extend(raw_data[0:2])  # \x80\x04
        i = 2

    while i <= last_stop:
        if raw_data[i] == 0x95 and i + 9 <= len(raw_data):
            # Skip FRAME opcode and its 8-byte argument
            i += 9
        else:
            cleaned.append(raw_data[i])
            i += 1

    cleaned_bytes = bytes(cleaned)
    print(f"  Cleaned data size (no frames): {len(cleaned_bytes)} (original to STOP: {last_stop + 1})")

    try:
        data = pickle.loads(cleaned_bytes)
        print(f"  Frame-stripped pickle.loads succeeded")
        return data
    except Exception as e:
        print(f"  Frame-stripped pickle.loads failed: {e}")

    try:
        data = dill.loads(cleaned_bytes)
        print(f"  Frame-stripped dill.loads succeeded")
        return data
    except Exception as e:
        print(f"  Frame-stripped dill.loads failed: {e}")

    raise RuntimeError("load_by_overriding_framing failed")


def load_with_bufferfix(filepath):
    """
    The issue might be that pickle reads the FRAME and tries to buffer
    exactly that many bytes, but there aren't enough. 
    We can work around this by providing a BytesIO with enough data.
    """
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) < 11:
        raise ValueError("Too small")

    declared_frame_len = struct.unpack('<Q', raw_data[3:11])[0]
    actual_remaining = len(raw_data) - 11
    shortfall = declared_frame_len - actual_remaining

    if shortfall > 0:
        print(f"  Padding {shortfall} bytes to match frame length")
        padded = raw_data + b'\x00' * shortfall
        try:
            data = dill.loads(padded)
            print(f"  Padded dill.loads succeeded")
            return data
        except Exception as e:
            print(f"  Padded dill.loads failed: {e}")

        try:
            data = pickle.loads(padded)
            print(f"  Padded pickle.loads succeeded")
            return data
        except Exception as e:
            print(f"  Padded pickle.loads failed: {e}")

        # Try padding with STOP opcode at the very end
        padded_with_stop = raw_data + b'\x00' * (shortfall - 1) + b'.'
        try:
            data = dill.loads(padded_with_stop)
            print(f"  Padded+STOP dill.loads succeeded")
            return data
        except Exception as e:
            print(f"  Padded+STOP dill.loads failed: {e}")

    raise RuntimeError("load_with_bufferfix failed")


def ultimate_load(filepath):
    """Try every possible method to load the file."""
    errors = []

    # Method 1: Direct load
    try:
        with open(filepath, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        errors.append(f"direct dill: {e}")
        print(f"  [1] Direct dill failed: {e}")

    # Method 2: Direct pickle
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        errors.append(f"direct pickle: {e}")
        print(f"  [2] Direct pickle failed: {e}")

    # Method 3: Fix frame length
    try:
        return load_with_extended_data(filepath)
    except Exception as e:
        errors.append(f"fix frame: {e}")
        print(f"  [3] Fix frame failed: {e}")

    # Method 4: Monkey patch
    try:
        return monkey_patch_and_load(filepath)
    except Exception as e:
        errors.append(f"monkey patch: {e}")
        print(f"  [4] Monkey patch failed: {e}")

    # Method 5: Strip frames
    try:
        return load_by_overriding_framing(filepath)
    except Exception as e:
        errors.append(f"strip frames: {e}")
        print(f"  [5] Strip frames failed: {e}")

    # Method 6: Buffer fix (pad to declared length)
    try:
        return load_with_bufferfix(filepath)
    except Exception as e:
        errors.append(f"buffer fix: {e}")
        print(f"  [6] Buffer fix failed: {e}")

    # Method 7: Load data with robust method
    try:
        return load_data_robust(filepath)
    except Exception as e:
        errors.append(f"robust: {e}")
        print(f"  [7] Robust failed: {e}")

    # Method 8: Original load_data
    try:
        return load_data(filepath)
    except Exception as e:
        errors.append(f"load_data: {e}")
        print(f"  [8] load_data failed: {e}")

    raise RuntimeError(f"All methods failed: {'; '.join(errors)}")


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
    inner_paths = []
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.pkl') and 'gradient_multi_shots' in f:
                fpath = os.path.join(data_dir, f)
                fsize = os.path.getsize(fpath)
                print(f"  Found: {f} ({fsize} bytes)")
                if ('parent_function' in f or 'parent_' in f) and fsize > 0:
                    inner_paths.append(fpath)

    print(f"\nOuter path: {outer_path} (exists: {os.path.exists(outer_path)})")
    print(f"Standard path: {standard_path} (exists: {os.path.exists(standard_path)})")
    print(f"Inner paths: {[os.path.basename(p) for p in inner_paths]}")

    # Determine which path to use
    paths_to_try = []
    if os.path.exists(standard_path):
        paths_to_try.append(standard_path)
    if os.path.exists(outer_path):
        paths_to_try.append(outer_path)

    # Also try any other gradient_multi_shots pkl files
    if os.path.isdir(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.pkl') and 'gradient_multi_shots' in f and 'parent' not in f:
                fpath = os.path.join(data_dir, f)
                if fpath not in paths_to_try and os.path.getsize(fpath) > 0:
                    paths_to_try.append(fpath)

    outer_data = None
    for path in paths_to_try:
        print(f"\nTrying to load: {path}")
        try:
            outer_data = ultimate_load(path)
            print(f"Successfully loaded from: {path}")
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue

    if outer_data is None:
        print("FAIL: Could not load outer data from any source")
        sys.exit(1)

    if not isinstance(outer_data, dict):
        if isinstance(outer_data, (list, tuple)) and len(outer_data) > 0 and isinstance(outer_data[0], dict):
            outer_data = outer_data[0]
        else:
            print(f"FAIL: Unexpected outer data type: {type(outer_data)}")
            sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"\nOuter func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    if outer_output is not None:
        print(f"Outer output type: {type(outer_output)}")
        if isinstance(outer_output, tuple):
            print(f"  Tuple length: {len(outer_output)}")
            for i, item in enumerate(outer_output):
                print(f"  [{i}] type={type(item).__name__}, ", end="")
                if isinstance(item, np.ndarray):
                    print(f"shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, (int, float)):
                    print(f"value={item}")
                else:
                    print(f"repr={repr(item)[:100]}")

    if inner_paths:
        # --- Scenario B: Factory/Closure pattern ---
        print("\n=== Scenario B: Factory/Closure pattern ===")

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
                inner_data = ultimate_load(inner_path)
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
        print("\n=== Scenario A: Simple function call ===")

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