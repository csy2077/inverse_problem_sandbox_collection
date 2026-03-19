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
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnCNN_DC1_sandbox/run_code/std_data/data_set_seeds.pkl'
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
        print("FAIL: Could not find outer data file (data_set_seeds.pkl)")
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

    print(f"Loaded outer data: func_name={outer_data.get('func_name', 'unknown')}")
    print(f"  args types: {[type(a).__name__ for a in outer_args]}")
    print(f"  kwargs keys: {list(outer_kwargs.keys())}")
    print(f"  output type: {type(outer_output).__name__}")

    # Phase 2: Execute function
    try:
        if len(inner_paths) > 0:
            # Scenario B: Factory/Closure pattern
            print("Detected Scenario B: Factory/Closure pattern")
            agent_operator = set_seeds(*outer_args, **outer_kwargs)

            if not callable(agent_operator):
                print(f"FAIL: set_seeds returned a non-callable: {type(agent_operator).__name__}")
                print("Falling back to Scenario A comparison...")
                # Fall back to Scenario A
                result = agent_operator
                expected = outer_output
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)

            # Process each inner path
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
                inner_output = inner_data.get('output', None)

                print(f"Loaded inner data: func_name={inner_data.get('func_name', 'unknown')}")

                try:
                    result = agent_operator(*inner_args, **inner_kwargs)
                except Exception as e:
                    print(f"FAIL: Error executing agent_operator with inner data: {e}")
                    traceback.print_exc()
                    sys.exit(1)

                expected = inner_output
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")

            print("TEST PASSED")
            sys.exit(0)

        else:
            # Scenario A: Simple function call
            print("Detected Scenario A: Simple function call")
            try:
                result = set_seeds(*outer_args, **outer_kwargs)
            except Exception as e:
                print(f"FAIL: Error executing set_seeds: {e}")
                traceback.print_exc()
                sys.exit(1)

            expected = outer_output

            # For set_seeds, the function returns None (it sets seeds as a side effect)
            # We verify the return value matches
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                # Additional verification: check that seeds were actually set
                # by verifying reproducibility
                try:
                    seed_val = outer_args[0] if len(outer_args) > 0 else outer_kwargs.get('seed', None)
                    if seed_val is not None:
                        # Re-set seeds and verify torch and numpy produce deterministic outputs
                        set_seeds(seed_val)
                        torch_val1 = torch.rand(1).item()
                        np_val1 = np.random.rand()

                        set_seeds(seed_val)
                        torch_val2 = torch.rand(1).item()
                        np_val2 = np.random.rand()

                        if torch_val1 != torch_val2:
                            print(f"FAIL: torch seeds not properly set. Got {torch_val1} vs {torch_val2}")
                            sys.exit(1)
                        if np_val1 != np_val2:
                            print(f"FAIL: numpy seeds not properly set. Got {np_val1} vs {np_val2}")
                            sys.exit(1)
                        print("Reproducibility check passed.")
                except Exception as e:
                    print(f"WARNING: Could not perform reproducibility check: {e}")

                print("TEST PASSED")
                sys.exit(0)

    except Exception as e:
        print(f"FAIL: Unexpected error during execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()