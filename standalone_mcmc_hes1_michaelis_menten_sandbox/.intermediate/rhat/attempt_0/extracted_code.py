import sys
import os
import dill
import numpy as np
import traceback

# Import the target function
from agent_rhat import rhat
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_hes1_michaelis_menten_sandbox/run_code/std_data/data_rhat.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found (data_rhat.pkl).")
        sys.exit(1)

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
        print(f"  args types: {[type(a).__name__ for a in outer_data.get('args', [])]}")
        print(f"  kwargs keys: {list(outer_data.get('kwargs', {}).keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure pattern.")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = rhat(*outer_args, **outer_kwargs)
            print(f"Phase 1: rhat() returned type: {type(agent_operator).__name__}")
        except Exception as e:
            print(f"FAIL: Phase 1 execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from rhat(), got {type(agent_operator).__name__}")
            sys.exit(1)

        # Phase 2: Execute with inner data
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
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: operator returned type: {type(result).__name__}")
            except Exception as e:
                print(f"FAIL: Phase 2 execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(inner_expected, result)
                if passed:
                    print(f"TEST PASSED (inner: {os.path.basename(inner_path)})")
                else:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"FAIL: Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call.")

        # Execute function
        try:
            result = rhat(*outer_args, **outer_kwargs)
            print(f"rhat() returned type: {type(result).__name__}")
        except Exception as e:
            print(f"FAIL: Execution failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Comparison
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAIL: Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()