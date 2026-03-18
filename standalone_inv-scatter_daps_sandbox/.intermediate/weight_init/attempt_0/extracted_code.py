import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import the target function
from agent_weight_init import weight_init

# Import verification utility
from verification_utils import recursive_check


def main():
    # All data paths provided
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_daps_sandbox/run_code/std_data/data_weight_init.pkl'
    ]

    # Separate outer (direct function data) and inner (parent_function / closure) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: Could not find outer data file (data_weight_init.pkl).")
        sys.exit(1)

    # --- Phase 1: Load outer data and reconstruct / run function ---
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Loaded outer data from: {outer_path}")
        print(f"[INFO] Keys in outer_data: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Determine scenario
    if len(inner_paths) > 0:
        # ---- Scenario B: Factory / Closure Pattern ----
        print("[INFO] Scenario B detected: Factory/Closure pattern.")

        # Phase 1: Create the operator
        try:
            agent_operator = weight_init(*outer_args, **outer_kwargs)
            print(f"[INFO] weight_init returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: weight_init(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable from weight_init, got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Loaded inner data [{idx}] from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data file [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception for inner [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception for inner [{idx}]: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for inner data [{idx}]: {msg}")
                all_passed = False
            else:
                print(f"[INFO] Inner data [{idx}] verification passed.")

        if not all_passed:
            sys.exit(1)

        print("TEST PASSED")
        sys.exit(0)

    else:
        # ---- Scenario A: Simple Function ----
        print("[INFO] Scenario A detected: Simple function call.")

        # Note: weight_init uses torch.rand / torch.randn which are stochastic.
        # The recorded output was generated with a specific random state.
        # We cannot reproduce it exactly. However, we can verify:
        # 1. The function runs without error with the given inputs.
        # 2. The output has the correct shape, dtype, and device.
        # 3. The output values are in the expected range for the given mode.

        try:
            result = weight_init(*outer_args, **outer_kwargs)
            print(f"[INFO] weight_init returned: {type(result)}")
        except Exception as e:
            print(f"FAIL: weight_init(*outer_args, **outer_kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # First try exact recursive_check (works if seeds were set)
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            # Since weight_init involves random sampling, exact match is unlikely.
            # Perform structural checks instead.
            print(f"[INFO] Exact match failed (expected for stochastic function): {msg}")
            print("[INFO] Performing structural verification instead...")

            structural_pass = True
            failure_reasons = []

            # Check type
            if type(expected) != type(result):
                structural_pass = False
                failure_reasons.append(f"Type mismatch: expected {type(expected)}, got {type(result)}")

            # Check shape
            if isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
                if expected.shape != result.shape:
                    structural_pass = False
                    failure_reasons.append(f"Shape mismatch: expected {expected.shape}, got {result.shape}")

                if expected.dtype != result.dtype:
                    structural_pass = False
                    failure_reasons.append(f"Dtype mismatch: expected {expected.dtype}, got {result.dtype}")

                # Verify result is finite
                if not torch.isfinite(result).all():
                    structural_pass = False
                    failure_reasons.append("Result contains non-finite values (inf/nan)")

                # Check that the scale is reasonable based on mode
                if len(outer_args) >= 2:
                    mode = outer_args[1]
                elif 'mode' in outer_kwargs:
                    mode = outer_kwargs['mode']
                else:
                    mode = None

                if len(outer_args) >= 3:
                    fan_in = outer_args[2]
                elif 'fan_in' in outer_kwargs:
                    fan_in = outer_kwargs['fan_in']
                else:
                    fan_in = None

                if len(outer_args) >= 4:
                    fan_out = outer_args[3]
                elif 'fan_out' in outer_kwargs:
                    fan_out = outer_kwargs['fan_out']
                else:
                    fan_out = None

                if mode and fan_in is not None:
                    # Verify the scale factor is correct by checking statistics
                    if mode == 'xavier_uniform' and fan_out is not None:
                        expected_bound = np.sqrt(6 / (fan_in + fan_out))
                        max_val = result.abs().max().item()
                        if max_val > expected_bound * 1.01:  # small tolerance
                            structural_pass = False
                            failure_reasons.append(
                                f"xavier_uniform: max abs value {max_val} exceeds bound {expected_bound}")

                    elif mode == 'kaiming_uniform':
                        expected_bound = np.sqrt(3 / fan_in)
                        max_val = result.abs().max().item()
                        if max_val > expected_bound * 1.01:
                            structural_pass = False
                            failure_reasons.append(
                                f"kaiming_uniform: max abs value {max_val} exceeds bound {expected_bound}")

                    elif mode == 'xavier_normal' and fan_out is not None:
                        expected_std = np.sqrt(2 / (fan_in + fan_out))
                        actual_std = result.std().item()
                        # Allow generous tolerance for random std
                        if abs(actual_std - expected_std) / expected_std > 0.5:
                            structural_pass = False
                            failure_reasons.append(
                                f"xavier_normal: std {actual_std} deviates too much from expected {expected_std}")

                    elif mode == 'kaiming_normal':
                        expected_std = np.sqrt(1 / fan_in)
                        actual_std = result.std().item()
                        if abs(actual_std - expected_std) / expected_std > 0.5:
                            structural_pass = False
                            failure_reasons.append(
                                f"kaiming_normal: std {actual_std} deviates too much from expected {expected_std}")

            elif isinstance(expected, np.ndarray) and isinstance(result, np.ndarray):
                if expected.shape != result.shape:
                    structural_pass = False
                    failure_reasons.append(f"Shape mismatch: expected {expected.shape}, got {result.shape}")

            if structural_pass:
                print("TEST PASSED (structural verification)")
                sys.exit(0)
            else:
                for reason in failure_reasons:
                    print(f"FAIL: {reason}")
                sys.exit(1)


if __name__ == '__main__':
    main()