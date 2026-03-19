import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Import target function
from agent_plot_trace import plot_trace
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/std_data/data_plot_trace.pkl'
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
        print("FAIL: No outer data file found for plot_trace.")
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
        print("Detected Scenario B (Factory/Closure pattern).")

        try:
            plt.close('all')
            agent_operator = plot_trace(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully created agent_operator.")
        except Exception as e:
            print(f"FAIL: Phase 1 - plot_trace() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Phase 1 - Expected callable operator, got {type(agent_operator)}")
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
                plt.close('all')
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Phase 2: Successfully executed agent_operator.")
            except Exception as e:
                print(f"FAIL: Phase 2 - agent_operator() raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for inner data: {os.path.basename(inner_path)}")
                else:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"FAIL: Verification raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function).")

        try:
            plt.close('all')
            result = plot_trace(*outer_args, **outer_kwargs)
            print("Phase 1: Successfully called plot_trace().")
        except Exception as e:
            print(f"FAIL: plot_trace() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Verification raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Clean up matplotlib figures
    plt.close('all')
    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()