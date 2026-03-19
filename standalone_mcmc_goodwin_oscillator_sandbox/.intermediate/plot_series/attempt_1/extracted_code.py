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

# Also inject into builtins as a fallback
try:
    from scipy.integrate import odeint
    import builtins
    builtins.odeint = odeint
except Exception:
    pass

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from agent_plot_series import plot_series
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/std_data/data_plot_series.pkl'
    ]

    # Classify paths into outer (direct function data) and inner (parent_function / closure data)
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
        # Patch odeint into the unpickle namespace
        from scipy.integrate import odeint as _odeint
        # Monkey-patch any module that might need it
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

    # After loading, patch odeint into any newly loaded modules
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

    # Also try to patch the problem object's model directly
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Try to fix the problem object's simulate method if needed
    try:
        from scipy.integrate import odeint as _odeint
        # problem is the second argument (index 1)
        problem = None
        if len(outer_args) > 1:
            problem = outer_args[1]
        elif 'problem' in outer_kwargs:
            problem = outer_kwargs['problem']

        if problem is not None:
            # Try to patch the model's module
            if hasattr(problem, '_model'):
                model = problem._model
                model_mod = sys.modules.get(type(model).__module__)
                if model_mod is not None and not hasattr(model_mod, 'odeint'):
                    model_mod.odeint = _odeint
                # Also try patching the simulate method's globals
                if hasattr(model, 'simulate'):
                    sim_func = model.simulate
                    if hasattr(sim_func, '__func__'):
                        sim_func.__func__.__globals__['odeint'] = _odeint
                    elif hasattr(sim_func, '__globals__'):
                        sim_func.__globals__['odeint'] = _odeint
            # Also patch problem.evaluate's globals
            if hasattr(problem, 'evaluate'):
                eval_func = problem.evaluate
                if hasattr(eval_func, '__func__'):
                    eval_func.__func__.__globals__['odeint'] = _odeint
                elif hasattr(eval_func, '__globals__'):
                    eval_func.__globals__['odeint'] = _odeint
    except Exception as e:
        print(f"  Warning: Could not patch odeint into problem object: {e}")

    # --- Determine scenario ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
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
            if len(outer_args) >= 3:
                outer_args = list(outer_args)
                outer_args[2] = temp_save_path
                outer_args = tuple(outer_args)
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

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"  recursive_check message: {msg}")
                print("  Attempting structural verification for matplotlib objects...")

                structural_pass = True
                structural_msg = ""

                if not isinstance(result, tuple):
                    structural_pass = False
                    structural_msg = f"Expected tuple, got {type(result)}"
                elif len(result) != 2:
                    structural_pass = False
                    structural_msg = f"Expected tuple of length 2, got length {len(result)}"
                else:
                    import matplotlib.pyplot as plt
                    import matplotlib.figure
                    fig_result, axes_result = result

                    if not isinstance(fig_result, matplotlib.figure.Figure):
                        structural_pass = False
                        structural_msg = f"First element not a Figure, got {type(fig_result)}"

                    if isinstance(axes_result, list):
                        for i, ax in enumerate(axes_result):
                            if not isinstance(ax, plt.Axes):
                                structural_pass = False
                                structural_msg = f"axes[{i}] not an Axes, got {type(ax)}"
                                break
                    elif not isinstance(axes_result, plt.Axes):
                        structural_pass = False
                        structural_msg = f"Second element not Axes or list of Axes, got {type(axes_result)}"

                    if isinstance(expected, tuple) and len(expected) == 2:
                        _, expected_axes = expected
                        if isinstance(expected_axes, list) and isinstance(axes_result, list):
                            if len(expected_axes) != len(axes_result):
                                structural_pass = False
                                structural_msg = f"Expected {len(expected_axes)} axes, got {len(axes_result)}"

                if structural_pass:
                    print("  Structural verification PASSED for matplotlib objects.")
                    passed = True
                else:
                    print(f"  Structural verification FAILED: {structural_msg}")
                    passed = False
                    msg = structural_msg

            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("PASS: Verification successful.")

        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Clean up temp file if it exists
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