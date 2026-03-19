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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_SiC_sandbox/run_code/std_data/data_TilePlot.pkl'
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

    # Load outer data
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
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = TilePlot(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created operator/closure.")
        except Exception as e:
            print(f"FAIL: Phase 1 - Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Phase 1 - Result is not callable, got {type(agent_operator)}")
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
                print("Phase 2: Successfully executed operator.")
            except Exception as e:
                print(f"FAIL: Phase 2 - Could not execute operator: {e}")
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
        print("Detected Scenario A: Simple function call")

        try:
            result = TilePlot(*outer_args, **outer_kwargs)
            print("Successfully executed TilePlot.")
        except Exception as e:
            print(f"FAIL: Could not execute TilePlot: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # For matplotlib objects, we need custom comparison since fig, im, ax
        # are not directly comparable with recursive_check in the usual sense.
        # We attempt recursive_check first, and if it fails on matplotlib objects,
        # we do a structural comparison.
        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                # The function returns (fig, im, ax) which are matplotlib objects.
                # These may not serialize/compare well. Let's do structural checks.
                print(f"recursive_check reported: {msg}")
                print("Attempting structural validation for matplotlib output...")

                # Validate that result is a tuple of 3 elements
                if not isinstance(result, tuple) or len(result) != 3:
                    print(f"FAIL: Expected tuple of 3, got {type(result)} of length {len(result) if isinstance(result, tuple) else 'N/A'}")
                    sys.exit(1)

                fig_result, im_result, ax_result = result

                # Check fig is a matplotlib Figure
                if not isinstance(fig_result, plt.Figure):
                    print(f"FAIL: First element should be matplotlib Figure, got {type(fig_result)}")
                    sys.exit(1)

                # Check im is a list of AxesImage
                if not isinstance(im_result, list):
                    print(f"FAIL: Second element should be list, got {type(im_result)}")
                    sys.exit(1)

                # Check ax is a numpy array
                if not isinstance(ax_result, np.ndarray):
                    print(f"FAIL: Third element should be numpy array, got {type(ax_result)}")
                    sys.exit(1)

                # If expected is also a tuple, compare structural properties
                if isinstance(expected, tuple) and len(expected) == 3:
                    _, expected_im, expected_ax = expected

                    # Compare number of images
                    if len(im_result) != len(expected_im):
                        print(f"FAIL: Expected {len(expected_im)} images, got {len(im_result)}")
                        sys.exit(1)

                    # Compare number of axes
                    if len(ax_result) != len(expected_ax):
                        print(f"FAIL: Expected {len(expected_ax)} axes, got {len(ax_result)}")
                        sys.exit(1)

                    # Compare image data arrays
                    for idx in range(len(im_result)):
                        try:
                            result_data = im_result[idx].get_array()
                            expected_data = expected_im[idx].get_array()
                            if result_data is not None and expected_data is not None:
                                result_arr = np.asarray(result_data)
                                expected_arr = np.asarray(expected_data)
                                if not np.allclose(result_arr, expected_arr, rtol=1e-5, atol=1e-8, equal_nan=True):
                                    print(f"FAIL: Image data mismatch at index {idx}")
                                    sys.exit(1)
                        except Exception as img_e:
                            print(f"Warning: Could not compare image data at index {idx}: {img_e}")

                print("Structural validation passed.")
                print("TEST PASSED")
                plt.close('all')
                sys.exit(0)
            else:
                print("TEST PASSED")
                plt.close('all')
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            plt.close('all')
            sys.exit(1)

    plt.close('all')
    sys.exit(0)


if __name__ == '__main__':
    main()