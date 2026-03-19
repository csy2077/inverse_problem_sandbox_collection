import sys
import os
import dill
import traceback

# Import the target function
from agent_is_url import is_url
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_is_url.pkl'
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
        print("FAIL: No outer data file found for is_url.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Loaded outer data: func_name={outer_data.get('func_name')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    print(f"  expected output type: {type(outer_output).__name__}")

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = is_url(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not create operator from is_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: is_url returned a non-callable: {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            print(f"Processing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Could not execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}")
                print(f"  Message: {msg}")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
                sys.exit(1)

            print(f"  Inner test passed for {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = is_url(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: Could not execute is_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()