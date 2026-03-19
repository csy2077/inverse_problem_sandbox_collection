import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_set_seeds import set_seeds
from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_PGMSU_DC1_sandbox/run_code/std_data/data_set_seeds.pkl'
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
        print("FAIL: No outer data file found for set_seeds.")
        sys.exit(1)

    # Phase 1: Load outer data and reconstruct
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

    print(f"Loaded outer data: func_name={outer_data.get('func_name')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    print(f"  output type: {type(outer_output).__name__}")

    # Phase 2: Execute function
    try:
        agent_result = set_seeds(*outer_args, **outer_kwargs)
    except Exception as e:
        print(f"FAIL: set_seeds raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Determine scenario
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")

        if not callable(agent_result):
            print(f"FAIL: Expected callable from set_seeds, got {type(agent_result).__name__}")
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
            expected = inner_data.get('output', None)

            print(f"Loaded inner data: func_name={inner_data.get('func_name')}")

            try:
                result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"  Inner test passed for {inner_data.get('func_name')}")

    else:
        # Scenario A: Simple function
        print("Detected Scenario A (Simple function)")

        result = agent_result
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)

    # Additional verification: ensure set_seeds actually sets seeds properly
    # by checking that torch and numpy produce deterministic results after calling
    try:
        seed_val = outer_args[0] if outer_args else outer_kwargs.get('seed', 42)
        
        set_seeds(seed_val)
        torch_val1 = torch.rand(5).tolist()
        np_val1 = np.random.rand(5).tolist()

        set_seeds(seed_val)
        torch_val2 = torch.rand(5).tolist()
        np_val2 = np.random.rand(5).tolist()

        if torch_val1 != torch_val2:
            print("FAIL: torch.manual_seed not set correctly - non-deterministic results")
            sys.exit(1)

        if np_val1 != np_val2:
            print("FAIL: np.random.seed not set correctly - non-deterministic results")
            sys.exit(1)

        print("  Determinism check passed")
    except Exception as e:
        print(f"WARNING: Could not verify determinism: {e}")
        traceback.print_exc()
        # Don't fail on this extra check

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()