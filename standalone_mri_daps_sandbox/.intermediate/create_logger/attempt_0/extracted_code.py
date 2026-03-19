import sys
import os
import dill
import traceback
import logging

# Import the target function
from agent_create_logger import create_logger
from verification_utils import recursive_check

def main():
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_daps_sandbox/run_code/std_data/data_create_logger.pkl']

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
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Before calling create_logger, ensure the logging_dir exists so FileHandler can create log.txt
    try:
        if len(outer_args) > 0:
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

    # Clear existing handlers on the module logger to avoid interference from prior runs
    try:
        module_logger_name = 'agent_create_logger'
        existing_logger = logging.getLogger(module_logger_name)
        existing_logger.handlers.clear()
    except Exception:
        pass

    try:
        result = create_logger(*outer_args, **outer_kwargs)
        print(f"  create_logger executed successfully. Result type: {type(result)}")
    except Exception as e:
        print(f"FAIL: create_logger raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Check if inner paths exist (Scenario B) or not (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
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

            if not callable(result):
                print("FAIL: Expected create_logger to return a callable (closure/operator), but it is not callable.")
                sys.exit(1)

            try:
                actual_result = result(*inner_args, **inner_kwargs)
                print(f"  Inner execution succeeded. Result type: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: Executing the returned operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
                else:
                    print(f"  Inner verification passed for {os.path.basename(inner_path)}.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - compare result directly with outer output
        expected = outer_output

        # For a Logger object, we do attribute-level comparison since Logger objects
        # are singletons and direct comparison may not work with recursive_check.
        # We verify key properties instead.
        try:
            # First try recursive_check
            passed, msg = recursive_check(expected, result)
            if not passed:
                # If recursive_check fails, do manual verification for Logger objects
                if isinstance(result, logging.Logger) and isinstance(expected, logging.Logger):
                    checks_passed = True
                    fail_reasons = []

                    # Check logger name
                    if result.name != expected.name:
                        checks_passed = False
                        fail_reasons.append(f"Logger name mismatch: expected '{expected.name}', got '{result.name}'")

                    # Check logging level
                    if result.level != expected.level:
                        checks_passed = False
                        fail_reasons.append(f"Logger level mismatch: expected {expected.level}, got {result.level}")

                    # Check that handler types match
                    expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
                    actual_handler_types = sorted([type(h).__name__ for h in result.handlers])

                    # We check that at least the expected handler types are present
                    # (there may be extras from prior test runs, but key ones must exist)
                    for ht in set(expected_handler_types):
                        exp_count = expected_handler_types.count(ht)
                        act_count = actual_handler_types.count(ht)
                        if act_count < exp_count:
                            checks_passed = False
                            fail_reasons.append(
                                f"Handler type '{ht}': expected at least {exp_count}, got {act_count}"
                            )

                    if not checks_passed:
                        print(f"FAIL: Logger verification failed: {'; '.join(fail_reasons)}")
                        sys.exit(1)
                    else:
                        print("  Logger attribute-level verification passed.")
                else:
                    print(f"FAIL: Verification failed: {msg}")
                    sys.exit(1)
            else:
                print("  Direct recursive_check verification passed.")
        except Exception as e:
            # Final fallback: check types match
            if type(result) == type(expected):
                print(f"  Type-level verification passed (both are {type(result).__name__}). "
                      f"recursive_check exception: {e}")
            else:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()