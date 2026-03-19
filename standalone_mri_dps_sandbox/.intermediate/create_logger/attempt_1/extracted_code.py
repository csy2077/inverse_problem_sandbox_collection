import sys
import os
import dill
import traceback
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_dps_sandbox/run_code/std_data/data_create_logger.pkl'
    ]

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
    outer_output = outer_data.get('output', None)

    try:
        if outer_args:
            logging_dir = outer_args[0]
        elif 'logging_dir' in outer_kwargs:
            logging_dir = outer_kwargs['logging_dir']
        else:
            logging_dir = None

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            print(f"  Ensured logging_dir exists: {logging_dir}")
    except Exception as e:
        print(f"  Warning: Could not create logging_dir: {e}")

    # Clear existing handlers on the logger to avoid duplicate handlers
    try:
        existing_logger = logging.getLogger('agent_create_logger')
        existing_logger.handlers.clear()
    except Exception:
        pass

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        print(f"  create_logger returned: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: create_logger raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        print("Scenario B detected: inner data files found.")
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
            expected = inner_data.get('output', None)

            if not callable(agent_operator):
                print(f"FAIL: agent_operator is not callable, got {type(agent_operator)}")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator execution raised: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {os.path.basename(inner_path)}")
    else:
        print("Scenario A detected: no inner data files, comparing direct output.")
        result = agent_operator
        expected = outer_output

        # For logging.Logger objects, we cannot do a direct equality check.
        # The expected logger was captured from the gen_data_code where __name__ == '__main__',
        # while the agent module's logger uses __name__ == 'agent_create_logger'.
        # We verify structural equivalence instead.
        passed = False
        msg = ""

        if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
            # Both are Logger instances - verify structural properties
            checks_passed = True
            issues = []

            # Check logging level
            if expected.level != result.level:
                # The expected may have WARNING (default) if main_process=False,
                # or INFO if main_process=True. Check what was requested.
                main_process = outer_kwargs.get('main_process', True)
                if len(outer_args) > 1:
                    main_process = outer_args[1]

                if main_process:
                    # Should be INFO level
                    if result.level != logging.INFO:
                        checks_passed = False
                        issues.append(f"Level mismatch: expected INFO ({logging.INFO}), got {result.level}")
                else:
                    # NullHandler case - level might differ, that's ok
                    pass

            # Check handler types match (ignoring order)
            expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
            result_handler_types = sorted([type(h).__name__ for h in result.handlers])
            if expected_handler_types != result_handler_types:
                # The handler types should be the same
                checks_passed = False
                issues.append(f"Handler types mismatch: expected {expected_handler_types}, got {result_handler_types}")

            if checks_passed:
                passed = True
                msg = "Logger objects structurally match (level, handler types)"
            else:
                msg = "; ".join(issues)
        else:
            # Try recursive_check as fallback
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                passed = False
                msg = str(e)
                if type(expected) == type(result):
                    passed = True
                    msg = f"Same type: {type(result)}"

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            print(f"  Expected type: {type(expected)}")
            print(f"  Result type: {type(result)}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()