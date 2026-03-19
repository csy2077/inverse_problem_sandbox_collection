import sys
import os
import dill
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_open_url import open_url
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_lgd_sandbox/run_code/std_data/data_open_url.pkl']

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
        print("ERROR: No outer data file found (data_open_url.pkl).")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        # Phase 1: Reconstruct operator
        try:
            agent_operator = open_url(*outer_args, **outer_kwargs)
            print(f"  Agent operator created: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Failed to create agent operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Agent operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Load inner data and execute
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"  Execution succeeded, result type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Failed to execute agent operator with inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Phase 3: Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {inner_path}")

        print("TEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function)")

        try:
            result = open_url(*outer_args, **outer_kwargs)
            print(f"  Function executed, result type: {type(result)}")
        except Exception as e:
            print(f"ERROR: Failed to execute open_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Handle special cases: if result is a file-like object (e.g., BufferedReader or BytesIO),
        # we need to compare the content or the filename depending on what was expected.
        # The function can return a string (filename), a file object, or BytesIO.
        # If expected is a file-like object, compare by reading content.
        # If expected is a string, compare directly.
        import io

        # If both are file-like objects, read and compare their contents
        result_to_compare = result
        expected_to_compare = expected

        if hasattr(expected, 'read') and hasattr(result, 'read'):
            try:
                expected_content = expected.read()
                result_content = result.read()
                # Reset positions if possible
                if hasattr(expected, 'seek'):
                    expected.seek(0)
                if hasattr(result, 'seek'):
                    result.seek(0)
                expected_to_compare = expected_content
                result_to_compare = result_content
            except Exception as e:
                print(f"WARNING: Could not read file objects for comparison: {e}")

        # If expected is a string (filename) and result is a string, compare directly
        # If expected is a file object and result is a file object, we already handled above

        try:
            passed, msg = recursive_check(expected_to_compare, result_to_compare)
        except Exception as e:
            print(f"ERROR: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()