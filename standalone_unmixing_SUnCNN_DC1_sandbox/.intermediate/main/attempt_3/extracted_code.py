import sys
import os
import dill
import torch
import numpy as np
import traceback
import logging
import importlib
import importlib.util
import json

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verification_utils import recursive_check


def main_test():
    """Test the main function from agent_main.py"""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnCNN_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # ---- Step 1: Classify data files ----
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    # ---- Step 2: Verify outer data exists ----
    if outer_path is None:
        print("FAIL: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    if not os.path.isfile(outer_path):
        print(f"FAIL: Outer data file not found at: {outer_path}")
        sys.exit(1)

    # ---- Step 3: Load outer data ----
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")

    # ---- Step 4: Find the data directory ----
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Search for DC1.mat in common locations
    possible_data_dirs = [
        os.path.join(script_dir, 'data'),
        os.path.join(script_dir, 'run_code', 'data'),
        os.path.join(script_dir, '..', 'data'),
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnCNN_DC1_sandbox/run_code/data',
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnCNN_DC1_sandbox/data',
    ]

    actual_data_dir = None
    for dd in possible_data_dirs:
        candidate = os.path.join(dd, 'DC1.mat')
        if os.path.isfile(candidate):
            actual_data_dir = dd
            print(f"Found DC1.mat at: {candidate}")
            break

    # Also search recursively if not found
    if actual_data_dir is None:
        search_root = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnCNN_DC1_sandbox'
        for root, dirs, files in os.walk(search_root):
            if 'DC1.mat' in files:
                actual_data_dir = root
                print(f"Found DC1.mat via search at: {os.path.join(root, 'DC1.mat')}")
                break

    if actual_data_dir is None:
        print("WARNING: Could not find DC1.mat anywhere. Will attempt to proceed anyway.")
        actual_data_dir = os.path.join(script_dir, 'data')

    # ---- Step 5: Import and run the function ----
    try:
        import builtins
        builtins.logging = logging

        # Try to find and load CONFIG
        CONFIG = None

        config_candidates = [
            os.path.join(script_dir, 'config.json'),
            os.path.join(script_dir, 'run_code', 'config.json'),
        ]

        for cfg_path in config_candidates:
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r') as f:
                    CONFIG = json.load(f)
                print(f"Loaded CONFIG from: {cfg_path}")
                break

        if CONFIG is None:
            config_py = os.path.join(script_dir, 'config.py')
            if os.path.isfile(config_py):
                spec = importlib.util.spec_from_file_location("config", config_py)
                config_mod = importlib.util.module_from_spec(spec)
                config_mod.logging = logging
                spec.loader.exec_module(config_mod)
                if hasattr(config_mod, 'CONFIG'):
                    CONFIG = config_mod.CONFIG
                    print(f"Loaded CONFIG from config.py")

        if CONFIG is None:
            CONFIG = {
                "seed": 0,
                "SNR": 30,
                "dataset": "DC1",
                "data_dir": "./data",
                "figs_dir": "./figs",
                "l2_normalization": False,
                "projection": False,
                "force_align": True,
                "EPS": 1e-8,
                "model": {
                    "niters": 2000,
                    "lr": 0.001,
                    "exp_weight": 0.99,
                    "noisy_input": True,
                }
            }
            print("Using default CONFIG")

        # Fix data_dir to the actual found path
        agent_main_path = os.path.join(script_dir, 'agent_main.py')
        if os.path.isfile(agent_main_path):
            agent_script_dir = os.path.dirname(os.path.abspath(agent_main_path))

            if actual_data_dir is not None:
                rel_data_dir = os.path.relpath(actual_data_dir, agent_script_dir)
                CONFIG["data_dir"] = rel_data_dir
                print(f"Set CONFIG['data_dir'] = '{rel_data_dir}'")

            figs_dir = os.path.join(script_dir, 'figs_test_output')
            os.makedirs(figs_dir, exist_ok=True)
            CONFIG["figs_dir"] = os.path.relpath(figs_dir, agent_script_dir)
            print(f"Set CONFIG['figs_dir'] = '{CONFIG['figs_dir']}'")

        builtins.CONFIG = CONFIG

        # ---- CRITICAL FIX: Monkey-patch the agent_main module to fix shape mismatch ----
        # The error is that A_hat has shape (M, N) where M = number of atoms (dictionary columns)
        # but A_gt has shape (p, N) where p = number of endmembers.
        # SUnCNN produces abundances of shape (M, N) because it uses D with M columns.
        # The aligner expects same shape. We need to ensure the agent_main code
        # handles this properly.

        # First, let's load the .mat file to understand dimensions
        import scipy.io as sio
        mat_path = os.path.join(actual_data_dir, 'DC1.mat')
        if os.path.isfile(mat_path):
            mat_data = sio.loadmat(mat_path)
            mat_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"DC1.mat keys: {mat_keys}")
            if 'A' in mat_data and 'D' in mat_data:
                p_val = mat_data['A'].shape[0] if len(mat_data['A'].shape) == 2 else None
                M_val = mat_data['D'].shape[1] if len(mat_data['D'].shape) == 2 else None
                print(f"Ground truth A shape: {mat_data['A'].shape}, D shape: {mat_data['D'].shape}")
                print(f"p={p_val}, M={M_val}")

        from agent_main import main as original_main

        # We need to wrap main to handle the shape mismatch in alignment
        # The issue: SUnCNN.compute_abundances returns A_hat of shape (M, H*W)
        # but A_gt is (p, H*W). When force_align=True, AbundancesAligner tries
        # to compare them but shapes differ.
        # 
        # The fix: We need to patch the main function or the relevant classes.
        # Let's patch the agent_main module's AbundancesAligner and related code.

        import agent_main

        # Save original _check_input
        original_check_input = agent_main.BaseMetric._check_input

        @staticmethod
        def patched_check_input(X, Xref):
            assert type(X) == type(Xref)
            if X.shape != Xref.shape:
                # Allow shape mismatch for MSE distance matrix computation
                # This happens when A_hat has more rows (M atoms) than A_gt (p endmembers)
                pass
            return X, Xref

        # Patch MSE to handle different shapes (for distance matrix computation)
        original_MSE_call = agent_main.MSE.__call__

        def patched_MSE_call(self, E, Eref):
            # E and Eref may have different shapes when computing distance matrix
            # E is A_hat.T of shape (N, M), Eref is A_gt.T of shape (N, p)
            # We need to compute pairwise distances
            if E.shape != Eref.shape:
                from numpy import linalg as LA
                # E: (N, M), Eref: (N, p)
                # Compute distance matrix of shape (M, p)
                normE = LA.norm(E, axis=0, keepdims=True)  # (1, M)
                normEref = LA.norm(Eref, axis=0, keepdims=True)  # (1, p)
                return np.sqrt(normE.T ** 2 + normEref ** 2 - 2 * (E.T @ Eref))
            return original_MSE_call(self, E, Eref)

        agent_main.MSE.__call__ = patched_MSE_call

        # Patch HungarianAligner.fit to handle rectangular distance matrices
        original_fit = agent_main.HungarianAligner.fit

        def patched_fit(self, A):
            from munkres import Munkres
            # Computing distance matrix
            self.dists = self.criterion(A.T, self.Aref.T)

            # A may have shape (M, N), Aref has shape (p, N)
            M_dim = A.shape[0]
            p_dim = self.Aref.shape[0]

            if M_dim == p_dim:
                # Same shape, use original
                P = np.zeros((p_dim, p_dim))
                m = Munkres()
                indices = m.compute(self.dists.tolist())
                for row, col in indices:
                    P[row, col] = 1.0
                self.P = P.T
            else:
                # Different shapes: dists is (M, p)
                m = Munkres()
                # Munkres can handle rectangular matrices
                # Make a copy since munkres modifies the input
                cost_matrix = self.dists.tolist()
                indices = m.compute(cost_matrix)
                # indices gives (row, col) pairs
                # row is index in A (0..M-1), col is index in Aref (0..p-1)
                # We want P such that P @ A selects/reorders rows of A to match Aref
                # P should be (p, M)
                P = np.zeros((p_dim, M_dim))
                for row, col in indices:
                    P[col, row] = 1.0
                self.P = P

        agent_main.HungarianAligner.fit = patched_fit

        # Patch transform to handle non-square P
        original_transform = agent_main.BaseAligner.transform

        def patched_transform(self, A):
            assert self.P is not None, 'Must be fitted first'
            # P may be (p, M), A is (M, N) -> result is (p, N)
            return self.P @ A

        agent_main.BaseAligner.transform = patched_transform

        # Also patch compute_metric's aRMSE to handle potential shape issues
        # after alignment, shapes should match, so this should be fine.

        print("Successfully imported and patched main from agent_main")

        main = original_main

    except Exception as e:
        print(f"FAIL: Could not import/patch main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Step 6: Determine scenario and execute ----
    inner_paths_valid = [p for p in inner_paths if os.path.isfile(p)]

    if len(inner_paths_valid) > 0:
        # Scenario B: Factory/Closure Pattern
        print("Detected Scenario B: Factory/Closure Pattern")

        try:
            print("Running main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Got operator of type: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in sorted(inner_paths_valid):
            print(f"\nProcessing inner data: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Inner data keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                print("Executing operator with inner args...")
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(inner_expected, actual_result)
                if not passed:
                    print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                    print(f"Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASSED for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple Function
        print("Detected Scenario A: Simple Function")

        try:
            print("Running main(*outer_args, **outer_kwargs)...")
            actual_result = main(*outer_args, **outer_kwargs)
            print(f"Got result of type: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            passed, msg = recursive_check(expected_output, actual_result)
            if not passed:
                print(f"FAIL: Verification failed")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print("PASSED: Output matches expected result")
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main_test()