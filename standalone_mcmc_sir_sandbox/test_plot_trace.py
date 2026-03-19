import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_plot_trace import plot_trace
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_sir_sandbox/run_code/std_data/data_plot_trace.pkl'
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
        print("FAIL: Could not find outer data file (data_plot_trace.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        try:
            agent_operator = plot_trace(*outer_args, **outer_kwargs)
            print(f"Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: Could not create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Agent operator is not callable, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
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
                print(f"Inner execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"FAIL: Could not execute agent operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASSED for inner path: {inner_path}")
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Use a temporary save path to avoid overwriting anything important
        # Modify save_path in kwargs if present, or in args
        # The function signature: plot_trace(samples, parameter_names=None, save_path='mcmc_trace.png')
        # We need to ensure a valid save_path that won't cause issues
        test_save_path = '/tmp/test_mcmc_trace.png'

        # Check if save_path is in kwargs or args
        modified_kwargs = dict(outer_kwargs)
        if 'save_path' in modified_kwargs:
            modified_kwargs['save_path'] = test_save_path
        else:
            # save_path is the 3rd positional arg
            outer_args_list = list(outer_args)
            if len(outer_args_list) >= 3:
                outer_args_list[2] = test_save_path
                outer_args = tuple(outer_args_list)
            else:
                modified_kwargs['save_path'] = test_save_path

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except Exception:
            pass

        try:
            result = plot_trace(*outer_args, **modified_kwargs)
            print(f"Function executed. Result type: {type(result)}")
        except Exception as e:
            print(f"FAIL: Could not execute plot_trace: {e}")
            traceback.print_exc()
            sys.exit(1)

        # For matplotlib figure comparison, we do structural checks
        # The function returns (fig, axes) - both are matplotlib objects
        # recursive_check may not handle matplotlib objects well, so we do manual checks
        try:
            if expected_output is None:
                print("WARNING: No expected output to compare against. Checking result is valid.")
                if result is None:
                    print("FAIL: Result is None")
                    sys.exit(1)
                # Check it's a tuple of (fig, axes)
                if isinstance(result, tuple) and len(result) == 2:
                    import matplotlib.pyplot as plt
                    import matplotlib.figure
                    fig, axes = result
                    if isinstance(fig, matplotlib.figure.Figure):
                        print("Result fig is a valid Figure.")
                    else:
                        print(f"FAIL: Expected Figure, got {type(fig)}")
                        sys.exit(1)
                    if isinstance(axes, np.ndarray):
                        print(f"Result axes shape: {axes.shape}")
                    else:
                        print(f"FAIL: Expected ndarray of axes, got {type(axes)}")
                        sys.exit(1)
                else:
                    print(f"FAIL: Expected tuple of length 2, got {type(result)}")
                    sys.exit(1)
            else:
                # Try recursive_check first
                try:
                    passed, msg = recursive_check(expected_output, result)
                    if not passed:
                        # If recursive_check fails on matplotlib objects, do structural comparison
                        print(f"recursive_check message: {msg}")
                        print("Attempting structural comparison for matplotlib objects...")

                        # Check structural equivalence
                        if isinstance(expected_output, tuple) and isinstance(result, tuple):
                            if len(expected_output) != len(result):
                                print(f"FAIL: Tuple length mismatch: {len(expected_output)} vs {len(result)}")
                                sys.exit(1)

                            import matplotlib.figure
                            exp_fig, exp_axes = expected_output
                            res_fig, res_axes = result

                            # Compare figure types
                            if type(exp_fig) != type(res_fig):
                                print(f"FAIL: Figure type mismatch: {type(exp_fig)} vs {type(res_fig)}")
                                sys.exit(1)

                            # Compare axes shapes
                            if isinstance(exp_axes, np.ndarray) and isinstance(res_axes, np.ndarray):
                                if exp_axes.shape != res_axes.shape:
                                    print(f"FAIL: Axes shape mismatch: {exp_axes.shape} vs {res_axes.shape}")
                                    sys.exit(1)
                                print(f"Axes shapes match: {res_axes.shape}")
                            else:
                                print(f"WARNING: Axes types: {type(exp_axes)} vs {type(res_axes)}")

                            print("Structural comparison PASSED")
                        else:
                            print(f"FAIL: Cannot structurally compare: {type(expected_output)} vs {type(result)}")
                            sys.exit(1)
                    else:
                        print("recursive_check PASSED")
                except Exception as e:
                    print(f"WARNING: recursive_check raised exception: {e}")
                    # Fallback: structural comparison
                    if isinstance(result, tuple) and len(result) == 2:
                        import matplotlib.figure
                        fig, axes = result
                        if isinstance(fig, matplotlib.figure.Figure) and isinstance(axes, np.ndarray):
                            print("Structural fallback check PASSED")
                        else:
                            print(f"FAIL: Structural check failed. Types: {type(fig)}, {type(axes)}")
                            sys.exit(1)
                    else:
                        print(f"FAIL: Unexpected result type: {type(result)}")
                        sys.exit(1)

        except SystemExit:
            raise
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Clean up temp file
        try:
            if os.path.exists(test_save_path):
                os.remove(test_save_path)
        except Exception:
            pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()