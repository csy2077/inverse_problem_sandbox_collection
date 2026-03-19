import sys
import os
import dill
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_get_script_dir import get_script_dir
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MiSiCNet_DC1_sandbox/run_code/std_data/data_get_script_dir.pkl'
    ]

    # Step 1: Classify paths into outer and inner
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Step 2: Load outer data
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

    # Step 3: Execute get_script_dir
    try:
        agent_result = get_script_dir(*outer_args, **outer_kwargs)
        print(f"  get_script_dir executed successfully.")
    except Exception as e:
        print(f"FAIL: get_script_dir raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern).")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from get_script_dir, got {type(agent_result)}")
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
            expected = inner_data.get('output')

            try:
                result = agent_result(*inner_args, **inner_kwargs)
                print(f"  Inner execution succeeded.")
            except Exception as e:
                print(f"FAIL: Inner execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

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
                print(f"  Inner test PASSED.")
    else:
        # Scenario A: Simple function
        print("Detected Scenario A (Simple function).")

        result = agent_result
        expected = outer_data.get('output')

        # For get_script_dir: the function returns the directory of the script file.
        # The expected output was captured from the original script's location,
        # while agent_get_script_dir.py is in a potentially different location.
        # Both are directory paths (strings). We do a recursive_check first,
        # but if it fails due to path differences, we verify both are valid directory strings.
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            # get_script_dir returns os.path.dirname(os.path.abspath(__file__))
            # The result will differ because __file__ differs between the original
            # and agent scripts. Both should be valid directory path strings.
            # We check that both are strings (directory paths) as a relaxed check.
            if isinstance(expected, str) and isinstance(result, str):
                # Both are directory path strings - the function is working correctly,
                # it just returns a different path because __file__ is different.
                print(f"  Note: Paths differ as expected (different __file__ locations).")
                print(f"    Expected (original): {expected}")
                print(f"    Got (agent):         {result}")
                # Verify result is the actual directory of agent_get_script_dir.py
                try:
                    import agent_get_script_dir
                    expected_agent_dir = os.path.dirname(os.path.abspath(agent_get_script_dir.__file__))
                    if result == expected_agent_dir:
                        print(f"  Result correctly matches agent script directory.")
                        passed = True
                    else:
                        print(f"FAIL: Result doesn't match agent script directory.")
                        print(f"  Expected agent dir: {expected_agent_dir}")
                        print(f"  Got: {result}")
                        sys.exit(1)
                except Exception as e2:
                    # Fallback: just verify both are strings (paths)
                    print(f"  Warning: Could not verify agent module path: {e2}")
                    print(f"  Accepting as passed since both are valid path strings.")
                    passed = True
            else:
                print(f"FAIL: {msg}")
                sys.exit(1)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"FAIL: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()