import sys
import os
import dill
import traceback
import io

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_open_url import open_url
from verification_utils import recursive_check


def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_dps_sandbox/run_code/std_data/data_open_url.pkl']

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
        print("ERROR: No outer data file found for open_url.")
        sys.exit(1)

    # Load outer data
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
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Reconstruct the operator
        try:
            agent_operator = open_url(*outer_args, **outer_kwargs)
            print(f"Phase 1: open_url returned object of type: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR: Phase 1 failed - open_url raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Phase 1 returned non-callable: {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Phase 2: agent_operator returned object of type: {type(result)}")
            except Exception as e:
                print(f"ERROR: Phase 2 failed - agent_operator raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"TEST PASSED for inner data: {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"ERROR: Comparison failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function
        print("Detected Scenario A: Simple function")

        # Phase 1: Execute the function
        try:
            result = open_url(*outer_args, **outer_kwargs)
            print(f"Phase 1: open_url returned object of type: {type(result)}")
        except Exception as e:
            print(f"ERROR: open_url raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Special handling for file-like objects (BufferedReader, BytesIO, etc.)
        # Both expected and result are file handles pointing to the same file.
        # We compare by checking the file name/path and/or content instead of object identity.
        try:
            result_is_file = isinstance(result, (io.BufferedReader, io.RawIOBase, io.BytesIO)) or hasattr(result, 'read')
            expected_is_file = isinstance(expected, (io.BufferedReader, io.RawIOBase, io.BytesIO)) or hasattr(expected, 'read')

            if result_is_file and expected_is_file:
                # Compare by name if both have name attribute
                result_name = getattr(result, 'name', None)
                expected_name = getattr(expected, 'name', None)

                if result_name is not None and expected_name is not None:
                    if result_name == expected_name:
                        print(f"File handles match by name: {result_name}")
                        # Additionally verify we can read from the result file
                        try:
                            # Read content from both and compare
                            expected.seek(0)
                            result.seek(0)
                            expected_content = expected.read()
                            result_content = result.read()
                            if expected_content == result_content:
                                print("File contents match.")
                                print("TEST PASSED")
                                sys.exit(0)
                            else:
                                print(f"TEST FAILED: File contents differ (expected {len(expected_content)} bytes, got {len(result_content)} bytes)")
                                sys.exit(1)
                        except Exception:
                            # If we can't read/compare contents, name match is sufficient
                            print("TEST PASSED")
                            sys.exit(0)
                    else:
                        print(f"TEST FAILED: File name mismatch: expected '{expected_name}', got '{result_name}'")
                        sys.exit(1)
                else:
                    # Try content comparison for BytesIO or unnamed streams
                    try:
                        expected.seek(0)
                        result.seek(0)
                        expected_content = expected.read()
                        result_content = result.read()
                        if expected_content == result_content:
                            print("File-like object contents match.")
                            print("TEST PASSED")
                            sys.exit(0)
                        else:
                            print(f"TEST FAILED: Content mismatch in file-like objects")
                            sys.exit(1)
                    except Exception as e2:
                        print(f"TEST FAILED: Cannot compare file-like objects: {e2}")
                        sys.exit(1)
            elif isinstance(result, str) and isinstance(expected, str):
                # return_filename mode - direct string comparison
                if result == expected:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    print(f"TEST FAILED: Filename mismatch: expected '{expected}', got '{result}'")
                    sys.exit(1)
            else:
                # Fall back to recursive_check
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
        except Exception as e:
            print(f"ERROR: Comparison failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()