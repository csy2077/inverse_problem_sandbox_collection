import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent_plot_trace import plot_trace
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/std_data/data_plot_trace.pkl'
    ]

    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for plot_trace.")
        sys.exit(1)

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

    # Scenario A: Simple function call
    print("Detected Scenario A (Simple function).")

    try:
        plt.close('all')
        # Prevent saving to file during test to avoid side effects
        # But keep original kwargs to match function behavior
        result = plot_trace(*outer_args, **outer_kwargs)
        print("Phase 1: Successfully called plot_trace().")
    except Exception as e:
        print(f"FAIL: plot_trace() raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    expected = outer_output

    # The function returns (fig, axes) - matplotlib objects can't be directly compared.
    # We need to do structural verification instead of direct object comparison.
    try:
        # Verify the result is a tuple of length 2
        if not isinstance(result, tuple):
            print(f"FAIL: Expected tuple result, got {type(result)}")
            sys.exit(1)

        if len(result) != 2:
            print(f"FAIL: Expected tuple of length 2, got length {len(result)}")
            sys.exit(1)

        result_fig, result_axes = result
        expected_fig, expected_axes = expected

        # Check that fig is a matplotlib Figure
        if not isinstance(result_fig, plt.Figure):
            print(f"FAIL: First element is not a Figure, got {type(result_fig)}")
            sys.exit(1)

        # Check axes shape matches
        if result_axes.shape != expected_axes.shape:
            print(f"FAIL: Axes shape mismatch: expected {expected_axes.shape}, got {result_axes.shape}")
            sys.exit(1)

        # Verify axes content: check labels match
        for i in range(result_axes.shape[0]):
            for j in range(result_axes.shape[1]):
                result_xlabel = result_axes[i, j].get_xlabel()
                expected_xlabel = expected_axes[i, j].get_xlabel()
                if result_xlabel != expected_xlabel:
                    print(f"FAIL: xlabel mismatch at axes[{i},{j}]: expected '{expected_xlabel}', got '{result_xlabel}'")
                    sys.exit(1)

                result_ylabel = result_axes[i, j].get_ylabel()
                expected_ylabel = expected_axes[i, j].get_ylabel()
                if result_ylabel != expected_ylabel:
                    print(f"FAIL: ylabel mismatch at axes[{i},{j}]: expected '{expected_ylabel}', got '{result_ylabel}'")
                    sys.exit(1)

        # Verify figure size matches
        expected_size = expected_fig.get_size_inches()
        result_size = result_fig.get_size_inches()
        if not np.allclose(expected_size, result_size, atol=1e-5):
            print(f"FAIL: Figure size mismatch: expected {expected_size}, got {result_size}")
            sys.exit(1)

        # Verify the number of lines/patches in each axes
        for i in range(result_axes.shape[0]):
            for j in range(result_axes.shape[1]):
                expected_n_lines = len(expected_axes[i, j].get_lines())
                result_n_lines = len(result_axes[i, j].get_lines())
                if expected_n_lines != result_n_lines:
                    print(f"FAIL: Line count mismatch at axes[{i},{j}]: expected {expected_n_lines}, got {result_n_lines}")
                    sys.exit(1)

                expected_n_patches = len(expected_axes[i, j].patches)
                result_n_patches = len(result_axes[i, j].patches)
                if expected_n_patches != result_n_patches:
                    print(f"FAIL: Patch count mismatch at axes[{i},{j}]: expected {expected_n_patches}, got {result_n_patches}")
                    sys.exit(1)

        # Verify line data matches numerically
        for i in range(result_axes.shape[0]):
            for j in range(result_axes.shape[1]):
                expected_lines = expected_axes[i, j].get_lines()
                result_lines = result_axes[i, j].get_lines()
                for k in range(len(expected_lines)):
                    exp_xdata = expected_lines[k].get_xdata()
                    res_xdata = result_lines[k].get_xdata()
                    exp_ydata = expected_lines[k].get_ydata()
                    res_ydata = result_lines[k].get_ydata()
                    if not np.allclose(np.array(exp_xdata, dtype=float), np.array(res_xdata, dtype=float), atol=1e-6, equal_nan=True):
                        print(f"FAIL: Line xdata mismatch at axes[{i},{j}] line {k}")
                        sys.exit(1)
                    if not np.allclose(np.array(exp_ydata, dtype=float), np.array(res_ydata, dtype=float), atol=1e-6, equal_nan=True):
                        print(f"FAIL: Line ydata mismatch at axes[{i},{j}] line {k}")
                        sys.exit(1)

        # Verify ylim on trace plots (column 1)
        for i in range(result_axes.shape[0]):
            expected_ylim = expected_axes[i, 1].get_ylim()
            result_ylim = result_axes[i, 1].get_ylim()
            if not np.allclose(expected_ylim, result_ylim, atol=1e-6):
                print(f"FAIL: ylim mismatch at axes[{i},1]: expected {expected_ylim}, got {result_ylim}")
                sys.exit(1)

        print("TEST PASSED")
    except SystemExit:
        raise
    except Exception as e:
        print(f"FAIL: Verification raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    plt.close('all')
    sys.exit(0)


if __name__ == '__main__':
    main()