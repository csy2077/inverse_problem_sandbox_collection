import sys
import os
import dill
import traceback
import numpy as np

# Ensure we can import the agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_save_reconstruction_plot import save_reconstruction_plot
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_save_reconstruction_plot.pkl'
    ]

    # Separate outer vs inner paths
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
    outer_output = outer_data.get('output', None)

    # The function save_reconstruction_plot saves a plot to a file and returns None.
    # We need to handle the output_path argument carefully:
    # - The original call may have used a specific path; we should use a temp path
    #   to avoid filesystem issues, then verify the function executes correctly.

    # Inspect args to find and potentially replace the output_path
    # Signature: save_reconstruction_plot(rec, output_path)
    # So args[0] = rec, args[1] = output_path (or in kwargs)

    # Create a temporary output path to avoid writing to the original location
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, 'test_reconstruction.png')

    # Replace the output_path in args or kwargs
    modified_args = list(outer_args)
    modified_kwargs = dict(outer_kwargs)

    if len(modified_args) >= 2:
        # args[1] is output_path
        modified_args[1] = temp_output_path
    elif 'output_path' in modified_kwargs:
        modified_kwargs['output_path'] = temp_output_path
    elif len(modified_args) == 1:
        # output_path might be positional but only rec was in args
        modified_kwargs['output_path'] = temp_output_path
    else:
        # Fallback: add as kwarg
        modified_kwargs['output_path'] = temp_output_path

    modified_args = tuple(modified_args)

    # Phase 2: Execute the function
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        try:
            print("Executing save_reconstruction_plot (outer call) to get operator...")
            agent_operator = save_reconstruction_plot(*modified_args, **modified_kwargs)
            print(f"  Outer call returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Outer call failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Verify agent_operator is callable
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
            expected = inner_data.get('output', None)

            try:
                print("Executing inner call (operator)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Inner call failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed: {msg}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        try:
            print("Executing save_reconstruction_plot (simple call)...")
            result = save_reconstruction_plot(*modified_args, **modified_kwargs)
            print(f"  Function returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: Function call failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Additional verification: check that the output file was created
        if os.path.exists(temp_output_path):
            print(f"  Output file created successfully: {temp_output_path}")
            file_size = os.path.getsize(temp_output_path)
            print(f"  File size: {file_size} bytes")
            if file_size == 0:
                print("FAIL: Output file is empty.")
                sys.exit(1)
        else:
            print(f"FAIL: Output file was not created at {temp_output_path}")
            sys.exit(1)

        # Compare return values (function returns None)
        try:
            passed, msg = recursive_check(expected, result)
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