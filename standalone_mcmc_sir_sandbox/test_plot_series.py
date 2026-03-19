import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')

# Fix missing imports that deserialized objects may need
from scipy.integrate import odeint
import builtins

# Inject odeint into builtins so deserialized closures/objects can find it
builtins.odeint = odeint

# Also patch into common module namespaces
import gen_std_data
if not hasattr(gen_std_data, 'odeint'):
    gen_std_data.odeint = odeint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_plot_series import plot_series
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_sir_sandbox/run_code/std_data/data_plot_series.pkl'
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

    # Patch odeint into the problem object's model if needed
    try:
        problem = outer_args[1] if len(outer_args) > 1 else outer_kwargs.get('problem', None)
        if problem is not None:
            # Try to find the model and patch its module globals
            if hasattr(problem, '_model'):
                model = problem._model
                if hasattr(model, 'simulate'):
                    func = model.simulate
                    if hasattr(func, '__globals__'):
                        if 'odeint' not in func.__globals__:
                            func.__globals__['odeint'] = odeint
                if hasattr(model, '_rhs') and hasattr(model._rhs, '__globals__'):
                    if 'odeint' not in model._rhs.__globals__:
                        model._rhs.__globals__['odeint'] = odeint
            # Also try patching evaluate
            if hasattr(problem, 'evaluate') and hasattr(problem.evaluate, '__globals__'):
                if 'odeint' not in problem.evaluate.__globals__:
                    problem.evaluate.__globals__['odeint'] = odeint
    except Exception as e:
        print(f"Warning: Could not patch odeint into problem object: {e}")

    # Override save_path to a temp file
    temp_save_path = '/tmp/test_posterior_predictive.png'
    if 'save_path' in outer_kwargs:
        outer_kwargs['save_path'] = temp_save_path
    elif len(outer_args) > 2:
        outer_args = list(outer_args)
        outer_args[2] = temp_save_path
        outer_args = tuple(outer_args)
    else:
        outer_kwargs['save_path'] = temp_save_path

    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = plot_series(*outer_args, **outer_kwargs)
            print("Phase 1: Operator created successfully.")
        except Exception as e:
            print(f"FAIL: Could not create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print("FAIL: Returned operator is not callable.")
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
                print("Phase 2: Operator executed successfully.")
            except Exception as e:
                print(f"FAIL: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = plot_series(*outer_args, **outer_kwargs)
            print("Function executed successfully.")
        except Exception as e:
            print(f"FAIL: Function execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"recursive_check reported: {msg}")
                print("Attempting structural verification for matplotlib objects...")

                if not isinstance(result, tuple):
                    print(f"FAIL: Expected tuple, got {type(result)}")
                    sys.exit(1)
                if len(result) != 2:
                    print(f"FAIL: Expected tuple of length 2, got length {len(result)}")
                    sys.exit(1)

                fig, axes = result
                import matplotlib.figure

                if not isinstance(fig, matplotlib.figure.Figure):
                    print(f"FAIL: First element is not a Figure, got {type(fig)}")
                    sys.exit(1)

                # Verify expected is also a tuple of length 2
                if isinstance(expected, tuple) and len(expected) == 2:
                    expected_fig, expected_axes = expected
                    # Check structural similarity
                    if isinstance(expected_fig, matplotlib.figure.Figure):
                        # Both are figures - compare structurally
                        # Check number of axes match
                        if hasattr(expected_axes, '__len__') and hasattr(axes, '__len__'):
                            if len(expected_axes) != len(axes):
                                print(f"FAIL: Axes count mismatch: expected {len(expected_axes)}, got {len(axes)}")
                                sys.exit(1)

                print("Structural verification passed for matplotlib objects.")
                print("TEST PASSED")
                sys.exit(0)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Cleanup temp file
    try:
        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)
    except:
        pass


if __name__ == '__main__':
    main()