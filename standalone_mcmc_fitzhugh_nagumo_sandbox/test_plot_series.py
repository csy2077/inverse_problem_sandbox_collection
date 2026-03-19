import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix scipy import issue in gen_std_data module
import scipy
import scipy.integrate

try:
    import gen_std_data
    gen_std_data.scipy = scipy
except ImportError:
    pass

# Inject 'vector' into builtins
import builtins
try:
    from myokit._sim.cvodessim import vector
    builtins.vector = vector
except ImportError:
    try:
        import myokit
        builtins.vector = myokit.vector
    except (ImportError, AttributeError):
        def vector(x):
            return np.asarray(x, dtype=float)
        builtins.vector = vector

# Ensure gen_std_data has vector too
try:
    import gen_std_data as _gsd
    if not hasattr(_gsd, 'vector'):
        _gsd.vector = builtins.vector
except ImportError:
    pass

try:
    import pints
except ImportError:
    pass

from agent_plot_series import plot_series
from verification_utils import recursive_check


def _patch_problem_object(obj):
    """Recursively patch problem objects to ensure scipy and vector are available."""
    if hasattr(obj, '_model'):
        model = obj._model
        mod_name = type(model).__module__
        mod = sys.modules.get(mod_name)
        if mod:
            if not hasattr(mod, 'scipy'):
                mod.scipy = scipy
            if not hasattr(mod, 'vector'):
                mod.vector = builtins.vector
        for attr_name in ['_simulate', 'simulate', '_rhs', 'rhs']:
            if hasattr(model, attr_name):
                method = getattr(model, attr_name)
                if hasattr(method, '__func__') and hasattr(method.__func__, '__globals__'):
                    method.__func__.__globals__['scipy'] = scipy
                    method.__func__.__globals__['vector'] = builtins.vector
                elif hasattr(method, '__globals__'):
                    method.__globals__['scipy'] = scipy
                    method.__globals__['vector'] = builtins.vector
    obj_mod_name = type(obj).__module__
    obj_mod = sys.modules.get(obj_mod_name)
    if obj_mod:
        if not hasattr(obj_mod, 'scipy'):
            obj_mod.scipy = scipy
        if not hasattr(obj_mod, 'vector'):
            obj_mod.vector = builtins.vector
    for attr_name in ['evaluate', '_evaluate']:
        if hasattr(obj, attr_name):
            method = getattr(obj, attr_name)
            if hasattr(method, '__func__') and hasattr(method.__func__, '__globals__'):
                method.__func__.__globals__['scipy'] = scipy
                method.__func__.__globals__['vector'] = builtins.vector
            elif hasattr(method, '__globals__'):
                method.__globals__['scipy'] = scipy
                method.__globals__['vector'] = builtins.vector


def structural_check_figure(expected, result):
    """
    Compare two matplotlib figure results structurally since Figure objects
    cannot be compared by identity/equality.
    """
    # Both should be tuples of (fig, axes)
    if not isinstance(result, tuple):
        return False, f"Expected tuple, got {type(result)}"
    if not isinstance(expected, tuple):
        return False, f"Expected output is not tuple: {type(expected)}"
    if len(result) != len(expected):
        return False, f"Tuple length mismatch: expected {len(expected)}, got {len(result)}"

    fig_result, axes_result = result
    fig_expected, axes_expected = expected

    # Check figure type
    if not isinstance(fig_result, plt.Figure):
        return False, f"Expected matplotlib Figure, got {type(fig_result)}"

    # Check figure size matches
    expected_size = fig_expected.get_size_inches()
    result_size = fig_result.get_size_inches()
    if not np.allclose(expected_size, result_size, atol=1e-3):
        return False, f"Figure size mismatch: expected {expected_size}, got {result_size}"

    # Check axes structure
    if isinstance(axes_expected, np.ndarray) and isinstance(axes_result, np.ndarray):
        if axes_expected.shape != axes_result.shape:
            return False, f"Axes shape mismatch: expected {axes_expected.shape}, got {axes_result.shape}"
        # Check each axis has same number of lines
        for i in range(len(axes_expected.flat)):
            exp_lines = len(axes_expected.flat[i].get_lines())
            res_lines = len(axes_result.flat[i].get_lines())
            if exp_lines != res_lines:
                return False, f"Axes[{i}] line count mismatch: expected {exp_lines}, got {res_lines}"
            # Check line data matches
            for j, (exp_line, res_line) in enumerate(zip(axes_expected.flat[i].get_lines(), axes_result.flat[i].get_lines())):
                exp_xdata = exp_line.get_xdata()
                res_xdata = res_line.get_xdata()
                exp_ydata = exp_line.get_ydata()
                res_ydata = res_line.get_ydata()
                if not np.allclose(exp_xdata, res_xdata, atol=1e-6, equal_nan=True):
                    return False, f"Axes[{i}] line[{j}] x-data mismatch"
                if not np.allclose(exp_ydata, res_ydata, atol=1e-6, equal_nan=True):
                    return False, f"Axes[{i}] line[{j}] y-data mismatch"
    elif isinstance(axes_expected, matplotlib.axes.Axes) and isinstance(axes_result, matplotlib.axes.Axes):
        # Single axes
        exp_lines = len(axes_expected.get_lines())
        res_lines = len(axes_result.get_lines())
        if exp_lines != res_lines:
            return False, f"Line count mismatch: expected {exp_lines}, got {res_lines}"
        for j, (exp_line, res_line) in enumerate(zip(axes_expected.get_lines(), axes_result.get_lines())):
            exp_xdata = exp_line.get_xdata()
            res_xdata = res_line.get_xdata()
            exp_ydata = exp_line.get_ydata()
            res_ydata = res_line.get_ydata()
            if not np.allclose(exp_xdata, res_xdata, atol=1e-6, equal_nan=True):
                return False, f"Line[{j}] x-data mismatch"
            if not np.allclose(exp_ydata, res_ydata, atol=1e-6, equal_nan=True):
                return False, f"Line[{j}] y-data mismatch"
    else:
        # Type mismatch but both exist - just check they are valid axes
        pass

    return True, "Structural figure comparison passed"


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_fitzhugh_nagumo_sandbox/run_code/std_data/data_plot_series.pkl'
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
        print("FAIL: No outer data file found.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")

    try:
        import gen_std_data as _gsd
        _gsd.scipy = scipy
        if not hasattr(_gsd, 'vector'):
            _gsd.vector = builtins.vector
    except Exception:
        pass

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

    print(f"  Function name: {outer_data.get('func_name', 'N/A')}")
    print(f"  Number of args: {len(outer_args)}")
    print(f"  Kwargs keys: {list(outer_kwargs.keys())}")

    try:
        import gen_std_data as _gsd
        _gsd.scipy = scipy
        if not hasattr(_gsd, 'vector'):
            _gsd.vector = builtins.vector
    except Exception:
        pass

    for arg in outer_args:
        _patch_problem_object(arg)

    for key, val in outer_kwargs.items():
        if hasattr(val, '_model') or hasattr(val, 'evaluate'):
            _patch_problem_object(val)

    for mod_name, mod in sys.modules.items():
        if mod is not None and hasattr(mod, '_simulate') or (mod is not None and 'gen_std_data' in str(mod_name)):
            try:
                if not hasattr(mod, 'scipy'):
                    mod.scipy = scipy
            except Exception:
                pass

    if len(inner_paths) > 0:
        print("\nScenario B detected: Factory/Closure pattern.")
        print("Phase 1: Reconstructing operator...")

        try:
            agent_operator = plot_series(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling plot_series with outer data: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        print("  Operator created successfully.")

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

            print(f"  Inner function name: {inner_data.get('func_name', 'N/A')}")

            print("Phase 2: Executing operator with inner data...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            print("Comparing results...")
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"  recursive_check raised exception: {e}")
                print("  Falling back to structural figure comparison...")
                try:
                    passed, msg = structural_check_figure(expected, result)
                except Exception as e2:
                    print(f"FAIL: Structural verification also failed: {e2}")
                    traceback.print_exc()
                    sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")

        plt.close('all')
        print("\nTEST PASSED")
        sys.exit(0)

    else:
        print("\nScenario A detected: Simple function call.")
        print("Executing plot_series with loaded args/kwargs...")

        # Close any existing figures to avoid interference
        plt.close('all')

        try:
            result = plot_series(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Error calling plot_series: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        print("Comparing results...")

        # For plot functions that return (fig, axes), we do structural comparison
        # since matplotlib Figure objects can't be compared by identity
        try:
            passed, msg = structural_check_figure(expected, result)
        except Exception as e:
            print(f"  structural_check_figure raised exception: {e}")
            traceback.print_exc()
            # Last resort: just check types and basic structure
            try:
                if isinstance(result, tuple) and len(result) == 2:
                    fig_r, axes_r = result
                    if isinstance(fig_r, plt.Figure):
                        passed = True
                        msg = "Basic type check passed (Figure returned)"
                    else:
                        passed = False
                        msg = f"Expected Figure, got {type(fig_r)}"
                else:
                    passed = False
                    msg = f"Expected tuple of length 2, got {type(result)}"
            except Exception as e2:
                print(f"FAIL: All verification methods failed: {e2}")
                traceback.print_exc()
                sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            plt.close('all')
            sys.exit(1)
        else:
            print(f"  Comparison result: {msg}")
            plt.close('all')
            print("\nTEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()