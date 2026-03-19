import sys
import os
import dill
import numpy as np
import traceback

from agent_getBufferSize import getBufferSize
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_SiC_sandbox/run_code/std_data/data_getBufferSize.pkl'
    ]

    # Classify paths into outer (direct) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for getBufferSize.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct/run the function
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    try:
        agent_result = getBufferSize(*outer_args, **outer_kwargs)
        print("Successfully called getBufferSize with outer data.")
    except Exception as e:
        print(f"FAIL: Error calling getBufferSize: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        # agent_result should be callable
        if not callable(agent_result):
            print(f"FAIL: Expected callable from getBufferSize, got {type(agent_result)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print("Successfully executed the operator with inner data.")
            except Exception as e:
                print(f"FAIL: Error executing operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASSED for inner path: {inner_path}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - result from Phase 1 IS the result
        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print("PASSED for outer data.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()