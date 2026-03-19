import sys
import os
import dill
import torch
import numpy as np
import traceback

# Import target function
from agent_weight_init import weight_init
from verification_utils import recursive_check


def main():
    # All data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_pnpdm_sandbox/run_code/std_data/data_weight_init.pkl'
    ]

    # Separate outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found for weight_init.")
        sys.exit(1)

    # Phase 1: Load outer data
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
    expected = outer_data.get('output')

    # Scenario A: Simple function call
    # weight_init uses torch.rand/torch.randn which are stochastic.
    # We cannot reproduce the exact random values, so we verify structural
    # properties: shape, dtype, and statistical bounds based on the init mode.
    print("Detected Scenario A: Simple function call")

    try:
        # Extract arguments to understand the mode
        import inspect
        sig = inspect.signature(weight_init)
        bound = sig.bind(*outer_args, **outer_kwargs)
        bound.apply_defaults()
        params = bound.arguments
        shape = params.get('shape')
        mode = params.get('mode')
        fan_in = params.get('fan_in')
        fan_out = params.get('fan_out')
        print(f"  shape={shape}, mode={mode}, fan_in={fan_in}, fan_out={fan_out}")
    except Exception as e:
        print(f"Warning: Could not inspect arguments: {e}")
        shape = None
        mode = None
        fan_in = None
        fan_out = None

    try:
        result = weight_init(*outer_args, **outer_kwargs)
        print(f"  Result type: {type(result)}")
    except Exception as e:
        print(f"FAIL: Could not execute weight_init: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Verify structural properties instead of exact values
    all_passed = True

    # Check type
    if not isinstance(result, type(expected)):
        print(f"FAIL: Type mismatch: expected {type(expected)}, got {type(result)}")
        sys.exit(1)
    print("  Type check: PASSED")

    # Check shape
    if hasattr(expected, 'shape') and hasattr(result, 'shape'):
        if expected.shape != result.shape:
            print(f"FAIL: Shape mismatch: expected {expected.shape}, got {result.shape}")
            sys.exit(1)
        print(f"  Shape check: PASSED ({result.shape})")

    # Check dtype
    if hasattr(expected, 'dtype') and hasattr(result, 'dtype'):
        if expected.dtype != result.dtype:
            print(f"FAIL: Dtype mismatch: expected {expected.dtype}, got {result.dtype}")
            sys.exit(1)
        print(f"  Dtype check: PASSED ({result.dtype})")

    # Check statistical bounds based on the initialization mode
    if mode is not None and isinstance(result, torch.Tensor):
        result_np = result.detach().cpu().numpy().astype(np.float64)
        expected_np = expected.detach().cpu().numpy().astype(np.float64)

        if mode == 'xavier_uniform':
            bound_val = np.sqrt(6.0 / (fan_in + fan_out))
            if result_np.max() > bound_val or result_np.min() < -bound_val:
                print(f"FAIL: xavier_uniform values out of bounds [-{bound_val}, {bound_val}]")
                sys.exit(1)
            if expected_np.max() > bound_val or expected_np.min() < -bound_val:
                print(f"Warning: expected values also out of theoretical bounds")
            print(f"  Bound check (xavier_uniform, bound={bound_val:.6f}): PASSED")

        elif mode == 'kaiming_uniform':
            bound_val = np.sqrt(3.0 / fan_in)
            if result_np.max() > bound_val or result_np.min() < -bound_val:
                print(f"FAIL: kaiming_uniform values out of bounds [-{bound_val}, {bound_val}]")
                sys.exit(1)
            print(f"  Bound check (kaiming_uniform, bound={bound_val:.6f}): PASSED")

        elif mode in ('xavier_normal', 'kaiming_normal'):
            # For normal distributions, check that the standard deviation is reasonable
            if mode == 'xavier_normal':
                expected_std = np.sqrt(2.0 / (fan_in + fan_out))
            else:
                expected_std = np.sqrt(1.0 / fan_in)

            actual_std = float(np.std(result_np))
            # Allow generous tolerance for stochastic check
            if result_np.size > 100:
                if actual_std > expected_std * 3.0 or actual_std < expected_std * 0.1:
                    print(f"FAIL: {mode} std deviation {actual_std:.6f} far from expected {expected_std:.6f}")
                    sys.exit(1)
            print(f"  Std check ({mode}, expected_std={expected_std:.6f}, actual_std={actual_std:.6f}): PASSED")

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()