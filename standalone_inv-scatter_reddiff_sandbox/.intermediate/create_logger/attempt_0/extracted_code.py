import sys
import os
import dill
import traceback
import logging

# Ensure the current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print("ERROR: No outer data file found for create_logger.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer function: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args types: {[type(a).__name__ for a in outer_args]}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")

    # The function creates a logger and needs a logging_dir that exists.
    # Ensure the logging directory exists so FileHandler can create log.txt
    if len(outer_args) > 0:
        logging_dir = outer_args[0]
        try:
            os.makedirs(logging_dir, exist_ok=True)
        except Exception:
            # If we can't create the original dir, use a temp dir
            import tempfile
            logging_dir = tempfile.mkdtemp()
            outer_args = (logging_dir,) + tuple(outer_args[1:])
    elif 'logging_dir' in outer_kwargs:
        logging_dir = outer_kwargs['logging_dir']
        try:
            os.makedirs(logging_dir, exist_ok=True)
        except Exception:
            import tempfile
            logging_dir = tempfile.mkdtemp()
            outer_kwargs['logging_dir'] = logging_dir

    # Clear any existing handlers on the module logger to avoid duplicates
    module_logger_name = 'agent_create_logger'
    existing_logger = logging.getLogger(module_logger_name)
    existing_logger.handlers.clear()

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to call create_logger: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"Agent operator type: {type(agent_operator).__name__}")

    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        for inner_path in inner_paths:
            print(f"\nProcessing inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner function: {inner_data.get('func_name', 'unknown')}")

            if not callable(agent_operator):
                print("ERROR: agent_operator is not callable for inner execution.")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator with inner args: {e}")
                traceback.print_exc()
                sys.exit(1)

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
                print(f"Inner test passed.")
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        result = agent_operator
        expected = outer_output

        # For Logger objects, we compare structurally rather than by identity
        # Check that both are Logger instances with matching properties
        if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
            # Compare logger properties
            checks_passed = True
            messages = []

            if expected.name != result.name:
                # The names might differ between agent module and gen module
                # Both should be module-level loggers, so we accept this
                print(f"Note: Logger names differ (expected='{expected.name}', got='{result.name}') - acceptable for cross-module test.")

            if expected.level != result.level:
                messages.append(f"Logger level mismatch: expected {expected.level}, got {result.level}")
                checks_passed = False

            # Check handler types match
            expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
            result_handler_types = sorted([type(h).__name__ for h in result.handlers])

            if expected_handler_types != result_handler_types:
                messages.append(
                    f"Handler types mismatch: expected {expected_handler_types}, got {result_handler_types}"
                )
                checks_passed = False

            if not checks_passed:
                print(f"TEST FAILED: {'; '.join(messages)}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        else:
            # Fall back to recursive_check
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"ERROR: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()