import sys
import os
import dill
import traceback

# Ensure the current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_mcgdiff_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print(f"FAIL: Could not load outer data file: {outer_path}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Loaded outer data: func_name={outer_data.get('func_name')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")

    # Ensure the logging directory exists if a path argument is provided
    try:
        if outer_args:
            logging_dir = outer_args[0]
            if isinstance(logging_dir, str):
                os.makedirs(logging_dir, exist_ok=True)
    except Exception:
        pass

    try:
        agent_operator = create_logger(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: create_logger raised an exception during construction.")
        traceback.print_exc()
        sys.exit(1)

    print(f"  agent_operator type: {type(agent_operator).__name__}")

    # Phase 2: Execution & Verification
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {inner_path}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Loaded inner data: func_name={inner_data.get('func_name')}")

            if not callable(agent_operator):
                print("FAIL: agent_operator is not callable but inner data exists (Scenario B).")
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
                print(f"FAIL: Verification failed for inner data {os.path.basename(inner_path)}.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {os.path.basename(inner_path)}.")
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        result = agent_operator
        expected = outer_output

        # For logger objects, we can't do a deep comparison directly.
        # We check structural equivalence: both should be logging.Logger instances
        # with matching names, levels, etc.
        import logging

        try:
            # First try recursive_check
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            # If recursive_check fails on Logger objects, do manual validation
            print(f"  recursive_check raised exception, falling back to manual check: {e}")
            passed = False
            msg = str(e)

        if not passed:
            # Manual check for Logger objects
            if isinstance(expected, logging.Logger) and isinstance(result, logging.Logger):
                checks_passed = True
                fail_reasons = []

                # Check logger name
                if expected.name != result.name:
                    checks_passed = False
                    fail_reasons.append(f"Logger name mismatch: expected={expected.name}, got={result.name}")

                # Check logging level
                if expected.level != result.level:
                    checks_passed = False
                    fail_reasons.append(f"Logger level mismatch: expected={expected.level}, got={result.level}")

                # Check handler types match
                expected_handler_types = sorted([type(h).__name__ for h in expected.handlers])
                result_handler_types = sorted([type(h).__name__ for h in result.handlers])

                # We check that the result has at least the expected handler types
                # (it might have more due to repeated test runs adding handlers)
                for ht in set(expected_handler_types):
                    exp_count = expected_handler_types.count(ht)
                    res_count = result_handler_types.count(ht)
                    if res_count < exp_count:
                        checks_passed = False
                        fail_reasons.append(
                            f"Handler type '{ht}' count mismatch: expected>={exp_count}, got={res_count}"
                        )

                if checks_passed:
                    passed = True
                    msg = "Manual logger comparison passed."
                else:
                    passed = False
                    msg = "; ".join(fail_reasons)
            else:
                # Not loggers, original failure stands
                pass

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"  Message: {msg}")
            print(f"  Expected type: {type(expected).__name__}")
            print(f"  Result type: {type(result).__name__}")
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()