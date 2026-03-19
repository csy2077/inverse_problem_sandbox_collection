import sys
import os
import dill
import traceback
import numpy as np

# Ensure we can import the agent module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_save_comparison_plot import save_comparison_plot
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_save_comparison_plot.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data
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

    # --- Phase 1: Load outer data ---
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

    # --- Determine scenario ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = save_comparison_plot(*outer_args, **outer_kwargs)
            print(f"  agent_operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create agent_operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        # The function save_comparison_plot returns None (it saves a plot to disk).
        # We need to handle the output_path carefully: use a temp path for the test
        # so we don't overwrite anything, but still verify the function runs correctly
        # and produces the expected output (which is None).

        # Modify output_path to a temporary location to avoid side effects
        import tempfile

        # Inspect args to find the output_path argument (3rd positional arg)
        modified_args = list(outer_args)
        if len(modified_args) >= 3:
            # Replace the output_path with a temp file
            original_output_path = modified_args[2]
            tmp_dir = tempfile.mkdtemp()
            tmp_output_path = os.path.join(tmp_dir, os.path.basename(str(original_output_path)))
            modified_args[2] = tmp_output_path
            print(f"  Redirecting output to temp path: {tmp_output_path}")

        modified_kwargs = dict(outer_kwargs)
        if 'output_path' in modified_kwargs:
            tmp_dir = tempfile.mkdtemp()
            tmp_output_path = os.path.join(tmp_dir, os.path.basename(str(modified_kwargs['output_path'])))
            modified_kwargs['output_path'] = tmp_output_path
            print(f"  Redirecting output to temp path: {tmp_output_path}")

        try:
            result = save_comparison_plot(*modified_args, **modified_kwargs)
            print(f"  Function executed successfully, result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute save_comparison_plot: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Phase 2: Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

        # Additional verification: check that the output file was actually created
        if len(modified_args) >= 3:
            check_path = modified_args[2]
        elif 'output_path' in modified_kwargs:
            check_path = modified_kwargs['output_path']
        else:
            check_path = None

        if check_path and os.path.exists(check_path):
            file_size = os.path.getsize(check_path)
            print(f"  Output file created successfully: {check_path} ({file_size} bytes)")
            # Clean up temp file
            try:
                os.remove(check_path)
                os.rmdir(os.path.dirname(check_path))
            except Exception:
                pass
        elif check_path:
            print(f"  WARNING: Output file was not created at: {check_path}")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()