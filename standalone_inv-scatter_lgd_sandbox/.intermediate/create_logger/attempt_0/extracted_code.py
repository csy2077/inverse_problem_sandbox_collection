import sys
import os
import dill
import traceback
import logging

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_lgd_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print("FAIL: No outer data file found for create_logger.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
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

    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")

    # Ensure the logging directory exists if needed (the function may try to create a file handler)
    if outer_args:
        logging_dir = outer_args[0]
        if isinstance(logging_dir, str):
            try:
                os.makedirs(logging_dir, exist_ok=True)
            except Exception:
                pass

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: create_logger raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"  agent_operator type: {type(agent_operator).__name__}")

    # Phase 2: Check if there are inner paths (Scenario B) or not (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        for inner_path in inner_paths:
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

            print(f"Inner data loaded from {os.path.basename(inner_path)}.")
            print(f"  inner args types: {[type(a).__name__ for a in inner_args]}")
            print(f"  inner kwargs keys: {list(inner_kwargs.keys())}")

            if not callable(agent_operator):
                print("FAIL: agent_operator is not callable for inner execution.")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {os.path.basename(inner_path)}.")
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        result = agent_operator
        expected = outer_output

        # For Logger objects, we do a structural comparison rather than direct equality
        # The logger returned should be of type logging.Logger
        if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
            # Compare key attributes of the logger
            checks_passed = True
            fail_messages = []

            if result.name != expected.name:
                checks_passed = False
                fail_messages.append(f"Logger name mismatch: expected={expected.name}, got={result.name}")

            if result.level != expected.level:
                checks_passed = False
                fail_messages.append(f"Logger level mismatch: expected={expected.level}, got={result.level}")

            if not checks_passed:
                print(f"FAIL: {'; '.join(fail_messages)}")
                sys.exit(1)
            else:
                print("  Logger structural comparison passed.")
        else:
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()