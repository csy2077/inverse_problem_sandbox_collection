import sys
import os
import dill
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_create_logger import create_logger
from verification_utils import recursive_check

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_diffpir_sandbox/run_code/std_data/data_create_logger.pkl'
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
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"[INFO] Outer args: {outer_args}")
    print(f"[INFO] Outer kwargs: {outer_kwargs}")

    # The function creates a logger which writes to a file. We need to ensure the logging_dir exists.
    # Extract logging_dir from args/kwargs
    try:
        # create_logger(logging_dir, main_process=True)
        # logging_dir is the first positional arg
        if outer_args:
            logging_dir = outer_args[0]
        else:
            logging_dir = outer_kwargs.get('logging_dir', None)

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

    # Phase 2: Check if there are inner paths (Scenario B) or not (Scenario A)
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected with {len(inner_paths)} inner data file(s).")
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                if not callable(agent_operator):
                    print(f"FAIL: agent_operator is not callable, got type {type(agent_operator)}")
                    sys.exit(1)
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Executing agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL (inner [{idx}]): {msg}")
                    sys.exit(1)
                else:
                    print(f"[INFO] Inner test [{idx}] passed.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function - the result from Phase 1 IS the result
        print("[INFO] Scenario A detected (simple function).")
        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print("[INFO] Outer test passed.")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)

if __name__ == '__main__':
    main()