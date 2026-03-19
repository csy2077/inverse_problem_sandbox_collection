import sys
import os
import dill
import numpy as np
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Import the target function
from agent_plot_trace import plot_trace
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_hes1_michaelis_menten_sandbox/run_code/std_data/data_plot_trace.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (standard_data_plot_trace.pkl or data_plot_trace.pkl).")
        sys.exit(1)

    # ---- Phase 1: Load outer data and reconstruct / run function ----
    try:
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # ---- Phase 2: Determine scenario and execute ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        try:
            print("Running plot_trace with outer args to get operator...")
            agent_operator = plot_trace(*outer_args, **outer_kwargs)
            plt.close('all')  # Clean up matplotlib figures
        except Exception as e:
            print(f"FAIL: Could not create operator from plot_trace: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                plt.close('all')
            except Exception as e:
                print(f"FAIL: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Inner data {os.path.basename(inner_path)} verified successfully.")
            except Exception as e:
                print(f"FAIL: Verification exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        try:
            print("Running plot_trace with outer args...")
            result = plot_trace(*outer_args, **outer_kwargs)
            plt.close('all')  # Clean up matplotlib figures
        except Exception as e:
            print(f"FAIL: plot_trace execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
            else:
                print("PASS: Output verified successfully.")
        except Exception as e:
            print(f"FAIL: Verification exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()