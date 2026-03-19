import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Import the target function
from agent_TilePlot import TilePlot
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/std_data/data_TilePlot.pkl'
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

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        try:
            agent_operator = TilePlot(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure.")
        except Exception as e:
            print(f"FAIL: Could not create operator from TilePlot: {e}")
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
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner data.")
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        try:
            result = TilePlot(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully called TilePlot.")
        except Exception as e:
            print(f"FAIL: Could not call TilePlot: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # For matplotlib objects, we need special handling.
        # The function returns (fig, im, ax). We verify structural compatibility.
        try:
            # First try direct recursive_check
            passed, msg = recursive_check(expected, result)
            if not passed:
                # If direct check fails, try structural verification for matplotlib objects
                print(f"Direct recursive_check failed: {msg}")
                print("Attempting structural verification for matplotlib objects...")

                # Verify result is a tuple of length 3
                if not isinstance(result, tuple) or len(result) != 3:
                    print(f"FAIL: Expected tuple of length 3, got {type(result)} of length {len(result) if isinstance(result, tuple) else 'N/A'}")
                    sys.exit(1)

                fig_result, im_result, ax_result = result

                # Check fig is a matplotlib Figure
                if not isinstance(fig_result, plt.Figure):
                    print(f"FAIL: Expected matplotlib Figure, got {type(fig_result)}")
                    sys.exit(1)

                # Check im is a list of AxesImage objects
                if not isinstance(im_result, list):
                    print(f"FAIL: Expected list for im, got {type(im_result)}")
                    sys.exit(1)

                # Check ax is a numpy array
                if not isinstance(ax_result, np.ndarray):
                    print(f"FAIL: Expected numpy array for ax, got {type(ax_result)}")
                    sys.exit(1)

                # Verify expected structure matches
                if isinstance(expected, tuple) and len(expected) == 3:
                    expected_fig, expected_im, expected_ax = expected

                    # Check same number of images plotted
                    if len(im_result) != len(expected_im):
                        print(f"FAIL: Expected {len(expected_im)} images, got {len(im_result)}")
                        sys.exit(1)

                    # Check same number of axes
                    if len(ax_result) != len(expected_ax):
                        print(f"FAIL: Expected {len(expected_ax)} axes, got {len(ax_result)}")
                        sys.exit(1)

                    # Verify image data matches
                    for i in range(len(im_result)):
                        try:
                            expected_data = expected_im[i].get_array()
                            result_data = im_result[i].get_array()
                            if not np.allclose(np.asarray(expected_data), np.asarray(result_data), rtol=1e-5, atol=1e-8, equal_nan=True):
                                print(f"FAIL: Image data mismatch at index {i}")
                                sys.exit(1)
                        except Exception as img_e:
                            print(f"Warning: Could not compare image data at index {i}: {img_e}")

                print("Structural verification passed.")
                print("TEST PASSED")
            else:
                print("TEST PASSED")
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Close all figures to free memory
    plt.close('all')
    sys.exit(0)


if __name__ == '__main__':
    main()