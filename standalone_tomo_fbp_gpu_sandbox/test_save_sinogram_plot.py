import sys
import os
import dill
import numpy as np
import traceback
import tempfile

# Import the target function
from agent_save_sinogram_plot import save_sinogram_plot
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_save_sinogram_plot.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (data_save_sinogram_plot.pkl)")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        try:
            agent_operator = save_sinogram_plot(*outer_args, **outer_kwargs)
            print("Phase 1: Operator created successfully.")
        except Exception as e:
            print(f"FAIL: Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        # The function saves a plot to a file and returns None.
        # We need to use a temporary output path to avoid file conflicts.
        
        try:
            # Make a copy of args as a list so we can modify output_path
            modified_args = list(outer_args)
            modified_kwargs = dict(outer_kwargs)
            
            # Try to use a temp file for the output path (second arg or 'output_path' kwarg)
            temp_dir = tempfile.mkdtemp()
            temp_output_path = os.path.join(temp_dir, 'test_sinogram.png')
            
            if len(modified_args) >= 2:
                modified_args[1] = temp_output_path
            elif 'output_path' in modified_kwargs:
                modified_kwargs['output_path'] = temp_output_path
            elif len(modified_args) == 1:
                modified_kwargs['output_path'] = temp_output_path
            
            modified_args = tuple(modified_args)
            
            result = save_sinogram_plot(*modified_args, **modified_kwargs)
            print("Phase 1: Function executed successfully.")
            
            # Verify the output file was created
            if os.path.exists(temp_output_path):
                print(f"  Output file created: {temp_output_path}")
                file_size = os.path.getsize(temp_output_path)
                print(f"  File size: {file_size} bytes")
                if file_size == 0:
                    print("FAIL: Output file is empty")
                    sys.exit(1)
            else:
                print("FAIL: Output file was not created")
                sys.exit(1)
            
        except Exception as e:
            print(f"FAIL: Could not execute function: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare result with expected output
        try:
            passed, msg = recursive_check(expected_output, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print(f"Verification passed: {msg}")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
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