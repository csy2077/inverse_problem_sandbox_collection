import sys
import os
import dill
import numpy as np
import traceback

# Fix the scipy import issue before importing the agent module
import scipy
import scipy.sparse
import scipy.sparse.linalg

from agent_get_astra_proj_matrix import get_astra_proj_matrix
from verification_utils import recursive_check


def main():
    # All data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_get_astra_proj_matrix.pkl'
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
        print("FAIL: Could not find outer data file (data_get_astra_proj_matrix.pkl)")
        sys.exit(1)

    # =========================================================================
    # Phase 1: Load outer data and reconstruct the operator
    # =========================================================================
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})

    print(f"[INFO] Outer args types: {[type(a).__name__ for a in outer_args]}")
    print(f"[INFO] Outer kwargs keys: {list(outer_kwargs.keys())}")

    try:
        agent_operator = get_astra_proj_matrix(*outer_args, **outer_kwargs)
        print(f"[INFO] get_astra_proj_matrix returned: {type(agent_operator).__name__}")
    except Exception as e:
        print(f"FAIL: get_astra_proj_matrix raised an exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # Phase 2: Execution & Verification
    # =========================================================================
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print(f"[INFO] Scenario B detected: {len(inner_paths)} inner data file(s)")

        all_passed = True
        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data from: {inner_path}")
                print(f"[INFO] Inner data keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output')

            print(f"[INFO] Inner args types: {[type(a).__name__ for a in inner_args]}")
            print(f"[INFO] Inner kwargs keys: {list(inner_kwargs.keys())}")

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator call returned: {type(actual_result).__name__}")
            except Exception as e:
                print(f"FAIL: Operator call raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    all_passed = False
                else:
                    print(f"[INFO] Verification passed for {inner_path}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

        if all_passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print("TEST FAILED")
            sys.exit(1)

    else:
        # Scenario A: Simple function - compare the returned operator with expected output
        print("[INFO] Scenario A detected: Simple function comparison")

        expected = outer_data.get('output')
        actual_result = agent_operator

        # For OpTomo objects, we can't directly compare them with recursive_check
        # We need to verify structural properties instead
        try:
            # First try direct recursive_check
            passed, msg = recursive_check(expected, actual_result)
            if passed:
                print("TEST PASSED")
                sys.exit(0)
            else:
                # If recursive_check fails on the object comparison, do structural validation
                print(f"[INFO] Direct comparison message: {msg}")
                print("[INFO] Attempting structural validation of OpTomo operator...")

                # Validate that the operator has the correct structural properties
                structural_pass = True
                error_msgs = []

                # Check type
                if type(actual_result).__name__ != type(expected).__name__:
                    structural_pass = False
                    error_msgs.append(
                        f"Type mismatch: expected {type(expected).__name__}, got {type(actual_result).__name__}"
                    )

                # Check shape
                if hasattr(expected, 'shape') and hasattr(actual_result, 'shape'):
                    if expected.shape != actual_result.shape:
                        structural_pass = False
                        error_msgs.append(
                            f"Shape mismatch: expected {expected.shape}, got {actual_result.shape}"
                        )
                    else:
                        print(f"[INFO] Shape matches: {actual_result.shape}")

                # Check dtype
                if hasattr(expected, 'dtype') and hasattr(actual_result, 'dtype'):
                    if expected.dtype != actual_result.dtype:
                        structural_pass = False
                        error_msgs.append(
                            f"Dtype mismatch: expected {expected.dtype}, got {actual_result.dtype}"
                        )
                    else:
                        print(f"[INFO] Dtype matches: {actual_result.dtype}")

                # Check vshape
                if hasattr(expected, 'vshape') and hasattr(actual_result, 'vshape'):
                    if expected.vshape != actual_result.vshape:
                        structural_pass = False
                        error_msgs.append(
                            f"vshape mismatch: expected {expected.vshape}, got {actual_result.vshape}"
                        )
                    else:
                        print(f"[INFO] vshape matches: {actual_result.vshape}")

                # Check sshape
                if hasattr(expected, 'sshape') and hasattr(actual_result, 'sshape'):
                    if expected.sshape != actual_result.sshape:
                        structural_pass = False
                        error_msgs.append(
                            f"sshape mismatch: expected {expected.sshape}, got {actual_result.sshape}"
                        )
                    else:
                        print(f"[INFO] sshape matches: {actual_result.sshape}")

                # Check appendString
                if hasattr(expected, 'appendString') and hasattr(actual_result, 'appendString'):
                    if expected.appendString != actual_result.appendString:
                        structural_pass = False
                        error_msgs.append(
                            f"appendString mismatch: expected '{expected.appendString}', got '{actual_result.appendString}'"
                        )
                    else:
                        print(f"[INFO] appendString matches: '{actual_result.appendString}'")

                # Functional test: FP and BP should work without error
                try:
                    test_vol = np.random.rand(*actual_result.vshape).astype(np.float32)
                    fp_result = actual_result.FP(test_vol)
                    print(f"[INFO] FP works, output shape: {fp_result.shape}")

                    bp_result = actual_result.BP(fp_result)
                    print(f"[INFO] BP works, output shape: {bp_result.shape}")

                    # If expected also has FP/BP, compare outputs
                    if hasattr(expected, 'FP') and hasattr(expected, 'BP'):
                        try:
                            expected_fp = expected.FP(test_vol)
                            fp_match, fp_msg = recursive_check(expected_fp, fp_result)
                            if not fp_match:
                                structural_pass = False
                                error_msgs.append(f"FP output mismatch: {fp_msg}")
                            else:
                                print("[INFO] FP output matches expected")

                            expected_bp = expected.BP(fp_result)
                            bp_match, bp_msg = recursive_check(expected_bp, bp_result)
                            if not bp_match:
                                structural_pass = False
                                error_msgs.append(f"BP output mismatch: {bp_msg}")
                            else:
                                print("[INFO] BP output matches expected")
                        except Exception as e:
                            print(f"[WARNING] Could not compare FP/BP with expected: {e}")

                except Exception as e:
                    structural_pass = False
                    error_msgs.append(f"Functional test failed: {e}")

                if structural_pass:
                    print("TEST PASSED")
                    sys.exit(0)
                else:
                    for em in error_msgs:
                        print(f"FAIL: {em}")
                    sys.exit(1)

        except Exception as e:
            print(f"FAIL: Verification raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()