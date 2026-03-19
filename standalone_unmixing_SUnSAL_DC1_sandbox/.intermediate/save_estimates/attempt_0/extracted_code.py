import sys
import os
import dill
import traceback

# Ensure the working directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_save_estimates import save_estimates
from verification_utils import recursive_check

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnSAL_DC1_sandbox/run_code/std_data/data_save_estimates.pkl'
    ]

    # Separate outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for save_estimates.")
        sys.exit(1)

    # ---------- Phase 1: Load outer data ----------
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # ---------- Determine scenario ----------
    if inner_paths:
        # ===== Scenario B: Factory/Closure Pattern =====
        print("Detected Scenario B (Factory/Closure pattern).")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = save_estimates(*outer_args, **outer_kwargs)
            print("Operator created successfully.")
        except Exception as e:
            print(f"FAIL: Error creating operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: save_estimates did not return a callable. Got: {type(agent_operator)}")
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
                print(f"FAIL: Error executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)

    else:
        # ===== Scenario A: Simple Function =====
        print("Detected Scenario A (Simple function).")

        # save_estimates returns None (it saves a file), so we need to handle that.
        # We'll use a temporary directory to avoid polluting the filesystem,
        # then verify the function runs without error and the output matches.

        # Check if the original output_dir argument exists in args; if so,
        # we may need to use a temp dir to actually write the file.
        import tempfile

        # The function signature: save_estimates(Ehat, Ahat, H, W, output_dir)
        # We need to potentially redirect output_dir to a temp directory
        # so the .mat file can be written successfully.

        modified_args = list(outer_args)
        modified_kwargs = dict(outer_kwargs)

        # Determine the position of output_dir (index 4 in positional args)
        temp_dir = None
        try:
            # Try to use a temp directory for output
            temp_dir = tempfile.mkdtemp(prefix="test_save_estimates_")

            # Replace output_dir in args or kwargs
            if len(modified_args) > 4:
                modified_args[4] = temp_dir
            elif 'output_dir' in modified_kwargs:
                modified_kwargs['output_dir'] = temp_dir
            else:
                # If we have exactly 4 positional args, append temp_dir
                if len(modified_args) == 4:
                    modified_args.append(temp_dir)

            modified_args = tuple(modified_args)
        except Exception as e:
            print(f"WARNING: Could not create temp directory: {e}")
            # Fall through and try with original args

        try:
            result = save_estimates(*modified_args, **modified_kwargs)
            print("Function executed successfully.")
        except Exception as e:
            print(f"FAIL: Error executing save_estimates: {e}")
            traceback.print_exc()
            sys.exit(1)

        # For a void function (returns None), verify:
        # 1. The output matches expected (both should be None)
        # 2. The .mat file was actually created
        expected = outer_output

        # Check that the file was created
        if temp_dir is not None:
            mat_file = os.path.join(temp_dir, "estimates.mat")
            if os.path.exists(mat_file):
                print(f"Verified: estimates.mat created at {mat_file}")
                # Optionally verify contents
                try:
                    import scipy.io as sio
                    loaded = sio.loadmat(mat_file)
                    if 'E' in loaded and 'A' in loaded:
                        print("Verified: .mat file contains 'E' and 'A' keys.")
                    else:
                        print(f"WARNING: .mat file keys: {list(loaded.keys())}")
                except Exception as e:
                    print(f"WARNING: Could not verify .mat contents: {e}")
            else:
                print(f"FAIL: estimates.mat was NOT created at {mat_file}")
                sys.exit(1)

            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        # Compare result vs expected output
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
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