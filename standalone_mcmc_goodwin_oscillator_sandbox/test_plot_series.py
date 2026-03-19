import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix missing imports in the gen_std_data module before anything else
try:
    from scipy.integrate import odeint
    import gen_std_data
    if not hasattr(gen_std_data, 'odeint'):
        gen_std_data.odeint = odeint
except Exception:
    pass

try:
    from scipy.integrate import odeint
    import builtins
    builtins.odeint = odeint
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure

from agent_plot_series import plot_series
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/std_data/data_plot_series.pkl'
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
        print("FAIL: No outer data file found for plot_series.")
        sys.exit(1)

    # --- Phase 1: Load outer data ---
    try:
        from scipy.integrate import odeint as _odeint
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, '__dict__'):
                try:
                    src = getattr(mod, '__file__', '') or ''
                    if 'gen_std_data' in src or 'gen_data' in src:
                        if not hasattr(mod, 'odeint'):
                            mod.odeint = _odeint
                except Exception:
                    pass

        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    # Patch odeint into newly loaded modules
    try:
        from scipy.integrate import odeint as _odeint
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, '__dict__'):
                try:
                    src = getattr(mod, '__file__', '') or ''
                    if 'gen_std_data' in src or 'gen_data' in src:
                        if not hasattr(mod, 'odeint'):
                            mod.odeint = _odeint
                except Exception:
                    pass
    except Exception:
        pass

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Patch odeint into problem object
    try:
        from scipy.integrate import odeint as _odeint
        problem = None
        if len(outer_args) > 1:
            problem = outer_args[1]
        elif 'problem' in outer_kwargs:
            problem = outer_kwargs['problem']

        if problem is not None:
            if hasattr(problem, '_model'):
                model = problem._model
                model_mod = sys.modules.get(type(model).__module__)
                if model_mod is not None and not hasattr(model_mod, 'odeint'):
                    model_mod.odeint = _odeint
                if hasattr(model, 'simulate'):
                    sim_func = model.simulate
                    if hasattr(sim_func, '__func__'):
                        sim_func.__func__.__globals__['odeint'] = _odeint
                    elif hasattr(sim_func, '__globals__'):
                        sim_func.__globals__['odeint'] = _odeint
            if hasattr(problem, 'evaluate'):
                eval_func = problem.evaluate
                if hasattr(eval_func, '__func__'):
                    eval_func.__func__.__globals__['odeint'] = _odeint
                elif hasattr(eval_func, '__globals__'):
                    eval_func.__globals__['odeint'] = _odeint
    except Exception as e:
        print(f"  Warning: Could not patch odeint into problem object: {e}")

    if len(inner_paths) > 0:
        # Scenario B
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = plot_series(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully called plot_series to get operator.")
        except Exception as e:
            print(f"FAIL: Error calling plot_series (Phase 1): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from plot_series, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed operator with inner data.")
            except Exception as e:
                print(f"FAIL: Error executing operator (Phase 2): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        modified_kwargs = dict(outer_kwargs)
        temp_save_path = '/tmp/test_posterior_predictive.png'
        if 'save_path' in modified_kwargs:
            modified_kwargs['save_path'] = temp_save_path
        else:
            outer_args_list = list(outer_args)
            if len(outer_args_list) >= 3:
                outer_args_list[2] = temp_save_path
                outer_args = tuple(outer_args_list)
            else:
                modified_kwargs['save_path'] = temp_save_path

        try:
            result = plot_series(*outer_args, **modified_kwargs)
            print("Successfully called plot_series.")
        except Exception as e:
            print(f"FAIL: Error calling plot_series: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # For matplotlib objects, recursive_check will fail on direct comparison.
        # We do structural verification instead.
        try:
            structural_pass = True
            structural_msg = ""

            # Check result is a tuple of length 2
            if not isinstance(result, tuple):
                structural_pass = False
                structural_msg = f"Expected tuple, got {type(result)}"
            elif len(result) != 2:
                structural_pass = False
                structural_msg = f"Expected tuple of length 2, got length {len(result)}"
            else:
                fig_result, axes_result = result

                # Check fig
                if not isinstance(fig_result, matplotlib.figure.Figure):
                    structural_pass = False
                    structural_msg = f"First element not a Figure, got {type(fig_result)}"

                # Check axes - can be list, numpy array, or single Axes
                if structural_pass:
                    axes_list = None
                    if isinstance(axes_result, np.ndarray):
                        axes_list = list(axes_result.flatten())
                    elif isinstance(axes_result, list):
                        axes_list = axes_result
                    elif isinstance(axes_result, plt.Axes):
                        axes_list = [axes_result]
                    else:
                        structural_pass = False
                        structural_msg = f"Second element not Axes, list, or ndarray of Axes, got {type(axes_result)}"

                    if axes_list is not None:
                        for i, ax in enumerate(axes_list):
                            if not isinstance(ax, plt.Axes):
                                structural_pass = False
                                structural_msg = f"axes[{i}] not an Axes, got {type(ax)}"
                                break

                # Check expected structure matches
                if structural_pass and isinstance(expected, tuple) and len(expected) == 2:
                    _, expected_axes = expected
                    expected_axes_list = None
                    if isinstance(expected_axes, np.ndarray):
                        expected_axes_list = list(expected_axes.flatten())
                    elif isinstance(expected_axes, list):
                        expected_axes_list = expected_axes
                    elif isinstance(expected_axes, plt.Axes):
                        expected_axes_list = [expected_axes]

                    if expected_axes_list is not None and axes_list is not None:
                        if len(expected_axes_list) != len(axes_list):
                            structural_pass = False
                            structural_msg = f"Expected {len(expected_axes_list)} axes, got {len(axes_list)}"

                # Verify the output file was created
                if structural_pass and os.path.exists(temp_save_path):
                    file_size = os.path.getsize(temp_save_path)
                    if file_size < 100:
                        structural_pass = False
                        structural_msg = f"Output file too small: {file_size} bytes"

            if structural_pass:
                print("Structural verification PASSED for matplotlib objects.")
            else:
                print(f"FAIL: Structural verification FAILED: {structural_msg}")
                sys.exit(1)

        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Clean up temp file
    try:
        temp_files = ['/tmp/test_posterior_predictive.png']
        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)
    except Exception:
        pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()