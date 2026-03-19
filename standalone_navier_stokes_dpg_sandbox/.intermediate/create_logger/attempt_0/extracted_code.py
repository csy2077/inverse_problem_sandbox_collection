import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging
import tempfile

from agent_create_logger import create_logger
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_dpg_sandbox/run_code/std_data/data_create_logger.pkl'
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
    expected_output = outer_data.get('output', None)

    # The function creates a logger that may write to a file.
    # We need to ensure the logging_dir exists. The first arg is logging_dir.
    # Create a temporary directory to avoid filesystem issues if the original dir doesn't exist.
    original_logging_dir = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('logging_dir', None)

    # Use a temp directory if the original doesn't exist
    temp_dir = None
    modified_args = list(outer_args)
    modified_kwargs = dict(outer_kwargs)

    if original_logging_dir and not os.path.exists(original_logging_dir):
        temp_dir = tempfile.mkdtemp()
        if len(modified_args) > 0:
            modified_args[0] = temp_dir
        elif 'logging_dir' in modified_kwargs:
            modified_kwargs['logging_dir'] = temp_dir

    # Clear existing handlers on the module logger to avoid accumulation
    logger_name = 'agent_create_logger'
    existing_logger = logging.getLogger(logger_name)
    existing_logger.handlers.clear()

    try:
        agent_operator = create_logger(*modified_args, **modified_kwargs)
    except Exception as e:
        print(f"FAIL: create_logger raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check if inner paths exist (Scenario B) or not (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator from create_logger, got {type(agent_operator)}")
            sys.exit(1)

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
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Executing agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data {inner_path}: {msg}")
                sys.exit(1)
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        result = agent_operator

        # For a logger object, we do a structural comparison rather than exact match
        # The logger is a logging.Logger instance; compare key properties
        try:
            # First try recursive_check directly
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            # If recursive_check fails on Logger objects, do manual verification
            try:
                # Both should be Logger instances
                if isinstance(expected_output, logging.Logger) and isinstance(result, logging.Logger):
                    # Check logger name matches
                    checks_pass = True
                    fail_msgs = []

                    if expected_output.name != result.name:
                        checks_pass = False
                        fail_msgs.append(f"Logger name mismatch: expected={expected_output.name}, got={result.name}")

                    if expected_output.level != result.level:
                        checks_pass = False
                        fail_msgs.append(f"Logger level mismatch: expected={expected_output.level}, got={result.level}")

                    # Check handler types match
                    expected_handler_types = sorted([type(h).__name__ for h in expected_output.handlers])
                    result_handler_types = sorted([type(h).__name__ for h in result.handlers])

                    if expected_handler_types != result_handler_types:
                        checks_pass = False
                        fail_msgs.append(
                            f"Handler types mismatch: expected={expected_handler_types}, got={result_handler_types}"
                        )

                    if checks_pass:
                        passed = True
                        msg = ""
                    else:
                        passed = False
                        msg = "; ".join(fail_msgs)
                else:
                    passed = False
                    msg = f"Type mismatch: expected {type(expected_output)}, got {type(result)}"
            except Exception as e2:
                print(f"FAIL: Manual verification also failed: {e2}")
                traceback.print_exc()
                sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed: {msg}")
            sys.exit(1)

    # Cleanup temp directory if created
    if temp_dir:
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()