import sys
import os
import dill
import traceback

# Ensure the script's directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_rel_path import rel_path
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_S2WSU_DC1_sandbox/run_code/std_data/data_rel_path.pkl'
    ]

    # --- Step 1: Classify data files ---
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file (data_rel_path.pkl) found.")
        sys.exit(1)

    # --- Step 2: Load outer data ---
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

    # --- Step 3: Execute rel_path with outer args ---
    try:
        agent_result = rel_path(*outer_args, **outer_kwargs)
        print(f"Phase 1: rel_path executed successfully.")
        print(f"  Result type: {type(agent_result).__name__}")
    except Exception as e:
        print(f"FAIL: rel_path(*args, **kwargs) raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Step 4: Determine scenario ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"Scenario B detected: {len(inner_paths)} inner data file(s) found.")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from rel_path, got {type(agent_result).__name__}")
            sys.exit(1)

        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Phase 2 [{idx}]: operator executed successfully.")
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"  Check [{idx}]: PASSED")

        if not all_passed:
            sys.exit(1)
        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function
        print("Scenario A detected: Simple function call.")

        expected = outer_data.get('output')
        result = agent_result

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()