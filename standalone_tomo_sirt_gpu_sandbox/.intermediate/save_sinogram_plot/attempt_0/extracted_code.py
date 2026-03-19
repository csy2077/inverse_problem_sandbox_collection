import sys
import os
import dill
import traceback
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_save_sinogram_plot import save_sinogram_plot
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_save_sinogram_plot.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct operator ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Since save_sinogram_plot saves a file to disk and returns None,
    # we need to handle the output_path carefully.
    # We'll use a temporary output path to avoid overwriting any existing files.
    import tempfile

    # The function signature is save_sinogram_plot(sinogram, output_path)
    # We need to redirect the output_path to a temp location
    # args[0] = sinogram, args[1] = output_path (or in kwargs)

    # Prepare a temporary output path
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, 'test_sinogram_output.png')

    # Modify the args to use the temp output path
    modified_args = list(outer_args)
    modified_kwargs = dict(outer_kwargs)

    # Determine where output_path is (positional or keyword)
    if len(modified_args) >= 2:
        # output_path is the second positional arg
        original_output_path = modified_args[1]
        modified_args[1] = temp_output_path
    elif 'output_path' in modified_kwargs:
        original_output_path = modified_kwargs['output_path']
        modified_kwargs['output_path'] = temp_output_path
    else:
        # Fallback: just try with original args
        original_output_path = None

    modified_args = tuple(modified_args)

    # ---- Scenario determination ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = save_sinogram_plot(*modified_args, **modified_kwargs)
            print(f"  Phase 1 result type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Error during Phase 1 (operator construction): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error during Phase 2 (operator execution): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner test passed: {msg}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = save_sinogram_plot(*modified_args, **modified_kwargs)
            print(f"  Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Error during function execution: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Additional verification: check that the output file was actually created
        if os.path.exists(temp_output_path):
            print(f"  Output file created successfully: {temp_output_path}")
            file_size = os.path.getsize(temp_output_path)
            print(f"  Output file size: {file_size} bytes")
            if file_size == 0:
                print("FAIL: Output file is empty (0 bytes)")
                sys.exit(1)
        else:
            print(f"FAIL: Output file was not created at: {temp_output_path}")
            sys.exit(1)

        # Compare the return value (should be None for this function)
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"  Return value verification passed: {msg}")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp files
    try:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()