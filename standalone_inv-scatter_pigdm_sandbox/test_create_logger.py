import sys
import os
import dill
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pigdm_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Clear any existing handlers on the logger to avoid accumulation from previous runs
    # The function uses logging.getLogger(__name__) which in agent_create_logger context
    # will be 'agent_create_logger'. We need to clear it before calling.
    try:
        target_logger = logging.getLogger('agent_create_logger')
        target_logger.handlers.clear()
        target_logger.setLevel(logging.WARNING)  # Reset to default
    except Exception:
        pass

    try:
        if outer_args:
            logging_dir = outer_args[0]
        elif 'logging_dir' in outer_kwargs:
            logging_dir = outer_kwargs['logging_dir']
        else:
            logging_dir = None

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            print(f"[INFO] Ensured logging directory exists: {logging_dir}")
    except Exception as e:
        print(f"[WARN] Could not create logging directory: {e}")

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
        print(f"[INFO] create_logger returned: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: create_logger raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    if inner_paths:
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            if not callable(agent_operator):
                print(f"FAIL: agent_operator is not callable, got type: {type(agent_operator)}")
                sys.exit(1)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] agent_operator returned: {type(result)}")
            except Exception as e:
                print(f"FAIL: agent_operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for inner path {inner_path}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Verification passed for inner path: {inner_path}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare the returned logger
        result = agent_operator
        expected = outer_output

        try:
            # Both should be Logger instances
            if not isinstance(result, logging.Logger):
                print(f"FAIL: Expected a logging.Logger instance, got {type(result)}")
                sys.exit(1)

            if not isinstance(expected, logging.Logger):
                print(f"FAIL: Expected output is not a logging.Logger, got {type(expected)}")
                sys.exit(1)

            # Determine what main_process was set to
            main_process = True
            if len(outer_args) > 1:
                main_process = outer_args[1]
            elif 'main_process' in outer_kwargs:
                main_process = outer_kwargs['main_process']

            if main_process:
                # When main_process=True, level should be INFO (20)
                if result.level != logging.INFO:
                    print(f"FAIL: Logger level mismatch: expected {logging.INFO} (INFO), got {result.level}")
                    sys.exit(1)
                print(f"[INFO] Logger level correct: {result.level} (INFO)")

                # Check handler types: should have StreamHandler and FileHandler
                result_handler_types = [type(h).__name__ for h in result.handlers]
                if 'StreamHandler' not in result_handler_types:
                    print(f"FAIL: Missing StreamHandler in result handlers: {result_handler_types}")
                    sys.exit(1)
                if 'FileHandler' not in result_handler_types:
                    print(f"FAIL: Missing FileHandler in result handlers: {result_handler_types}")
                    sys.exit(1)
                print(f"[INFO] Handler types correct: {result_handler_types}")
            else:
                # When main_process=False, should have NullHandler
                result_handler_types = [type(h).__name__ for h in result.handlers]
                if 'NullHandler' not in result_handler_types:
                    print(f"FAIL: Missing NullHandler in result handlers: {result_handler_types}")
                    sys.exit(1)
                print(f"[INFO] Handler types correct: {result_handler_types}")

            # Verify the logger is functional (can log without error)
            try:
                result.info("Test log message for verification")
            except Exception as e:
                print(f"FAIL: Logger cannot log: {e}")
                sys.exit(1)

            print("[INFO] Logger verification passed (type, level, handlers checked).")

        except SystemExit:
            raise
        except Exception as e:
            print(f"FAIL: Verification raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()