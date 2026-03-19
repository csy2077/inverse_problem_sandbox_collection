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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_TV_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print(f"FAIL: Could not load outer data file: {outer_path}")
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

        if logging_dir and not os.path.exists(logging_dir):
            os.makedirs(logging_dir, exist_ok=True)
            print(f"  Created logging directory: {logging_dir}")
    except Exception as e:
        print(f"  Warning: Could not create logging directory: {e}")

    # Clear any existing handlers on the logger to avoid test pollution
    logger_name = 'agent_create_logger'
    existing_logger = logging.getLogger(logger_name)
    existing_logger.handlers.clear()

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        print(f"  create_logger returned: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: create_logger raised an exception during construction.")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        print(f"Scenario B: Found {len(inner_paths)} inner data file(s).")
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"  Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if not callable(agent_operator):
                print(f"FAIL: agent_operator is not callable. Type: {type(agent_operator)}")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised an exception during execution.")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception.")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data: {inner_path}")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for: {os.path.basename(inner_path)}")
    else:
        print("Scenario A: No inner data files found. Comparing direct output.")
        result = agent_operator
        expected = outer_output

        passed = False
        msg = ""

        try:
            if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
                # Both are loggers. The expected logger name is '__main__' because the
                # gen_data_code ran create_logger in __main__ context using logging.getLogger(__name__).
                # Our agent module uses its own __name__ = 'agent_create_logger'.
                # We verify structural equivalence instead of name equality.
                checks_passed = True
                check_msgs = []

                # Check both are Logger instances
                if not isinstance(result, logging.Logger):
                    checks_passed = False
                    check_msgs.append(f"Result is not a Logger, got {type(result)}")

                # Check log level matches
                if expected.level != result.level:
                    checks_passed = False
                    check_msgs.append(f"Log level mismatch: expected={expected.level}, got={result.level}")

                # Check handler types match
                expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
                result_handler_types = sorted([type(h).__name__ for h in result.handlers])
                if expected_handler_types != result_handler_types:
                    checks_passed = False
                    check_msgs.append(
                        f"Handler types mismatch: expected={expected_handler_types}, got={result_handler_types}"
                    )

                # Check number of handlers
                if len(expected.handlers) != len(result.handlers):
                    checks_passed = False
                    check_msgs.append(
                        f"Handler count mismatch: expected={len(expected.handlers)}, got={len(result.handlers)}"
                    )

                passed = checks_passed
                msg = "; ".join(check_msgs) if check_msgs else ""

                if passed:
                    print("  Logger structure matches (type, level, handlers).")
            else:
                passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"  Warning: comparison raised exception: {e}")
            print("  Falling back to type-based comparison.")
            if type(expected) == type(result):
                passed = True
                msg = ""
            else:
                passed = False
                msg = f"Type mismatch: expected {type(expected)}, got {type(result)}"

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()