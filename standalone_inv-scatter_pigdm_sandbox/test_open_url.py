import sys
import os
import dill
import traceback

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pigdm_sandbox/run_code/std_data/data_open_url.pkl'
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
        print("FAIL: No outer data file found for open_url.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Run outer function to get the operator
        try:
            agent_operator = open_url(*outer_args, **outer_kwargs)
            print("  Successfully created agent_operator from open_url()")
        except Exception as e:
            print(f"FAIL: Error calling open_url with outer args: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        # Process each inner data file
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("  Successfully executed agent_operator with inner args")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Verification passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = open_url(*outer_args, **outer_kwargs)
            print("  Successfully called open_url()")
        except Exception as e:
            print(f"FAIL: Error calling open_url: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Handle special cases: if both result and expected are file-like objects (BytesIO or file),
        # compare their contents instead of the objects themselves
        try:
            import io

            # If expected is a string (filename) and result is a string, compare directly
            # If expected is bytes-like or file-like, read contents for comparison
            result_for_check = result
            expected_for_check = expected

            # If both are file-like (BytesIO or file objects), read and compare content
            if hasattr(expected, 'read') and hasattr(result, 'read'):
                expected_content = expected.read()
                result_content = result.read()
                # Reset positions
                if hasattr(expected, 'seek'):
                    expected.seek(0)
                if hasattr(result, 'seek'):
                    result.seek(0)
                expected_for_check = expected_content
                result_for_check = result_content
            elif isinstance(expected, str) and isinstance(result, str):
                # Both are filenames, compare as strings
                expected_for_check = expected
                result_for_check = result
            elif isinstance(expected, io.BytesIO) and isinstance(result, io.BytesIO):
                expected_for_check = expected.getvalue()
                result_for_check = result.getvalue()

            passed, msg = recursive_check(expected_for_check, result_for_check)
        except Exception as e:
            # Fallback: try direct comparison
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e2:
                print(f"FAIL: Error during verification: {e2}")
                traceback.print_exc()
                sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)
        else:
            print("  Verification passed.")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()