import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Ensure the current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_plot_series import plot_series
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/std_data/data_plot_series.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data files
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (data_plot_series.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern.")

        # Run outer function to get operator
        try:
            agent_operator = plot_series(*outer_args, **outer_kwargs)
            plt.close('all')  # Close any figures created
        except Exception as e:
            print(f"FAIL: Error running plot_series to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from plot_series, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Operator created successfully: {type(agent_operator)}")

        # Process each inner data file
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            # Execute operator with inner args
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                plt.close('all')
            except Exception as e:
                print(f"FAIL: Error executing operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: Error during recursive_check: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call.")

        # Run the function
        try:
            result = plot_series(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error running plot_series: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # For matplotlib figures, we need special handling
        # The output is (fig, axes) - check structural compatibility
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")

        # Compare results
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            # If recursive_check fails on matplotlib objects, do structural verification
            print(f"Warning: recursive_check raised exception: {e}")
            print("Attempting structural verification for matplotlib objects...")

            try:
                # Verify result is a tuple of (fig, axes)
                if not isinstance(result, tuple) or len(result) != 2:
                    print(f"FAIL: Expected tuple of length 2, got {type(result)}")
                    sys.exit(1)

                fig_result, axes_result = result

                if not isinstance(fig_result, plt.Figure):
                    print(f"FAIL: First element should be matplotlib Figure, got {type(fig_result)}")
                    sys.exit(1)

                # Check expected is also a tuple
                if isinstance(expected, tuple) and len(expected) == 2:
                    fig_expected, axes_expected = expected

                    # Compare figure sizes
                    result_size = fig_result.get_size_inches()
                    expected_size = fig_expected.get_size_inches()
                    if not np.allclose(result_size, expected_size, atol=1e-5):
                        print(f"FAIL: Figure sizes differ. Expected {expected_size}, got {result_size}")
                        sys.exit(1)

                    # Compare number of axes
                    result_axes_list = fig_result.get_axes()
                    expected_axes_list = fig_expected.get_axes()
                    if len(result_axes_list) != len(expected_axes_list):
                        print(f"FAIL: Number of axes differ. Expected {len(expected_axes_list)}, got {len(result_axes_list)}")
                        sys.exit(1)

                    # Compare number of lines in each axis
                    for i, (ax_r, ax_e) in enumerate(zip(result_axes_list, expected_axes_list)):
                        r_lines = len(ax_r.get_lines())
                        e_lines = len(ax_e.get_lines())
                        if r_lines != e_lines:
                            print(f"FAIL: Axis {i} line count differs. Expected {e_lines}, got {r_lines}")
                            sys.exit(1)

                        # Compare line data
                        for j, (line_r, line_e) in enumerate(zip(ax_r.get_lines(), ax_e.get_lines())):
                            xdata_r = np.array(line_r.get_xdata())
                            xdata_e = np.array(line_e.get_xdata())
                            ydata_r = np.array(line_r.get_ydata())
                            ydata_e = np.array(line_e.get_ydata())

                            if not np.allclose(xdata_r, xdata_e, atol=1e-5, equal_nan=True):
                                print(f"FAIL: Axis {i}, Line {j} x-data differs.")
                                sys.exit(1)
                            if not np.allclose(ydata_r, ydata_e, atol=1e-5, equal_nan=True):
                                print(f"FAIL: Axis {i}, Line {j} y-data differs.")
                                sys.exit(1)

                passed = True
                msg = "Structural verification passed for matplotlib objects."
                print(msg)

            except SystemExit:
                raise
            except Exception as e2:
                print(f"FAIL: Structural verification also failed: {e2}")
                traceback.print_exc()
                sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

        plt.close('all')

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()