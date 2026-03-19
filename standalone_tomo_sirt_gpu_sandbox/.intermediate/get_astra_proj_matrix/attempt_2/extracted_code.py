import sys
import os
import dill
import numpy as np
import traceback

# Must import scipy before importing the agent module
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Inject scipy into builtins so agent module can find it at class-definition time
import builtins
builtins.scipy = scipy

# Now safe to import the agent module
import agent_get_astra_proj_matrix
agent_get_astra_proj_matrix.scipy = scipy

from agent_get_astra_proj_matrix import get_astra_proj_matrix
from verification_utils import recursive_check


def main():
    # All data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_sirt_gpu_sandbox/run_code/std_data/data_get_astra_proj_matrix.pkl'
    ]

    # Classify paths into outer (direct function data) and inner (parent_function / operator execution data)
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_' in basename or 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # --- Phase 1: Load outer data and reconstruct operator ---
    if outer_path is None:
        print("FAIL: No outer data file found for get_astra_proj_matrix.")
        sys.exit(1)

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer function name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_operator = get_astra_proj_matrix(*outer_args, **outer_kwargs)
        print(f"Agent operator created successfully. Type: {type(agent_operator)}")
    except Exception as e:
        print(f"FAIL: get_astra_proj_matrix raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Phase 2: Execution & Verification ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"\nScenario B detected: {len(inner_paths)} inner data file(s) found.")
        all_passed = True

        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner Test {idx + 1}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data file: {e}")
                traceback.print_exc()
                all_passed = False
                continue

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner function name: {inner_data.get('func_name', 'N/A')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            try:
                func_name = inner_data.get('func_name', '')

                if hasattr(agent_operator, func_name) and callable(getattr(agent_operator, func_name)):
                    method = getattr(agent_operator, func_name)
                    method_args = inner_args[1:] if len(inner_args) > 0 else ()
                    print(f"Calling agent_operator.{func_name}() with {len(method_args)} args")
                    result = method(*method_args, **inner_kwargs)
                else:
                    print(f"Calling agent_operator directly with inner args")
                    result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Execution of inner function raised an exception: {e}")
                traceback.print_exc()
                all_passed = False
                continue

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed: {msg}")
                    all_passed = False
                else:
                    print(f"PASSED: Inner test {idx + 1} verification successful.")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                all_passed = False

        if not all_passed:
            print("\nTEST FAILED: One or more inner tests failed.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function - compare the operator itself against expected output
        print("\nScenario A detected: No inner data files. Comparing operator output directly.")

        result = agent_operator
        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed: {msg}")
                print(f"Expected type: {type(expected)}")
                print(f"Result type: {type(result)}")

                if hasattr(expected, 'shape') and hasattr(result, 'shape'):
                    print(f"Expected shape: {expected.shape}, Result shape: {result.shape}")
                if hasattr(expected, 'vshape') and hasattr(result, 'vshape'):
                    print(f"Expected vshape: {expected.vshape}, Result vshape: {result.vshape}")
                if hasattr(expected, 'sshape') and hasattr(result, 'sshape'):
                    print(f"Expected sshape: {expected.sshape}, Result sshape: {result.sshape}")

                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()

            # Fallback: manual comparison of key attributes for OpTomo objects
            print("\nAttempting fallback attribute-level comparison...")
            try:
                fallback_passed = True
                if hasattr(expected, 'shape') and hasattr(result, 'shape'):
                    if expected.shape != result.shape:
                        print(f"FAIL: Shape mismatch: expected {expected.shape}, got {result.shape}")
                        fallback_passed = False
                    else:
                        print(f"Shape match: {result.shape}")

                if hasattr(expected, 'vshape') and hasattr(result, 'vshape'):
                    if expected.vshape != result.vshape:
                        print(f"FAIL: vshape mismatch: expected {expected.vshape}, got {result.vshape}")
                        fallback_passed = False
                    else:
                        print(f"vshape match: {result.vshape}")

                if hasattr(expected, 'sshape') and hasattr(result, 'sshape'):
                    if expected.sshape != result.sshape:
                        print(f"FAIL: sshape mismatch: expected {expected.sshape}, got {result.sshape}")
                        fallback_passed = False
                    else:
                        print(f"sshape match: {result.sshape}")

                if hasattr(expected, 'appendString') and hasattr(result, 'appendString'):
                    if expected.appendString != result.appendString:
                        print(f"FAIL: appendString mismatch: expected '{expected.appendString}', got '{result.appendString}'")
                        fallback_passed = False
                    else:
                        print(f"appendString match: '{result.appendString}'")

                # Test forward projection with a random input
                if hasattr(expected, 'FP') and hasattr(result, 'FP'):
                    if hasattr(result, 'vshape'):
                        np.random.seed(42)
                        test_input = np.random.rand(*result.vshape).astype(np.float32)
                        expected_fp = expected.FP(test_input.copy())
                        result_fp = result.FP(test_input.copy())
                        fp_passed, fp_msg = recursive_check(expected_fp, result_fp)
                        if not fp_passed:
                            print(f"FAIL: FP output mismatch: {fp_msg}")
                            fallback_passed = False
                        else:
                            print("FP output match.")

                if fallback_passed:
                    print("TEST PASSED (via fallback comparison)")
                    sys.exit(0)
                else:
                    print("TEST FAILED (fallback comparison)")
                    sys.exit(1)
            except Exception as e2:
                print(f"FAIL: Fallback comparison also failed: {e2}")
                traceback.print_exc()
                sys.exit(1)


if __name__ == '__main__':
    main()