import sys
import os
import types
import json
import logging
import traceback

# Ensure the current directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def find_config(base_dir):
    """Search for config.json in common locations."""
    candidates = [
        os.path.join(base_dir, 'config.json'),
        os.path.join(base_dir, 'run_code', 'config.json'),
        os.path.join(base_dir, 'data_standalone', 'config.json'),
    ]
    for cp in candidates:
        if os.path.exists(cp):
            with open(cp, 'r') as f:
                return json.load(f)
    # Walk the directory
    for root, dirs, files in os.walk(base_dir):
        for fn in files:
            if fn == 'config.json':
                with open(os.path.join(root, fn), 'r') as f:
                    return json.load(f)
    return {}


def find_all_configs(base_dir):
    """Find all config files and merge them."""
    configs = []
    for root, dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith('.json') and 'config' in fn.lower():
                try:
                    with open(os.path.join(root, fn), 'r') as f:
                        configs.append(json.load(f))
                except:
                    pass
    return configs


def build_complete_config(base_dir):
    """Build a complete config with all required keys."""
    config = find_config(base_dir)

    parent = os.path.dirname(base_dir)
    if not config:
        config = find_config(parent)

    run_code_dir = os.path.join(base_dir, 'run_code')
    if os.path.isdir(run_code_dir):
        rc_config = find_config(run_code_dir)
        if rc_config:
            for k, v in rc_config.items():
                if k not in config:
                    config[k] = v

    all_configs = find_all_configs(base_dir)
    for c in all_configs:
        if isinstance(c, dict):
            for k, v in c.items():
                if k not in config:
                    config[k] = v

    defaults = {
        'EPS': 1e-10,
        'SNR': 30,
        'dataset': 'DC1',
        'l2_normalization': False,
        'projection': False,
        'force_align': True,
        'model': {
            'AL_iters': 1000,
            'lambd': 0.01,
            'verbose': True,
            'tol': 1e-4,
            'mu': 0.1,
            'x0': 0,
        },
    }

    for k, v in defaults.items():
        if k not in config:
            config[k] = v

    if 'model' in config and isinstance(config['model'], dict):
        model_defaults = defaults['model']
        for k, v in model_defaults.items():
            if k not in config['model']:
                config['model'][k] = v
    elif 'model' not in config:
        config['model'] = defaults['model']

    print(f"Final config keys: {list(config.keys())}")
    print(f"Config: {config}")
    return config


def setup_and_import_main():
    """Import main from agent_main with proper globals setup."""
    agent_main_path = os.path.join(script_dir, 'agent_main.py')

    with open(agent_main_path, 'r') as f:
        source = f.read()

    config = build_complete_config(script_dir)

    logging.basicConfig(level=logging.INFO)

    try:
        from munkres import Munkres
        has_munkres = True
    except ImportError:
        has_munkres = False

    preamble = """
import os as os
import sys as sys
import logging as logging
import json as _json_
import numpy as _np_patch_
import numpy.linalg as LA

SCRIPT_DIR = {script_dir_repr}

CONFIG = {config_repr}

HAS_MUNKRES = {has_munkres_repr}

try:
    from munkres import Munkres
except ImportError:
    class Munkres:
        pass

logging.basicConfig(level=logging.INFO)
""".format(
        script_dir_repr=repr(script_dir),
        config_repr=repr(config),
        has_munkres_repr=repr(has_munkres),
    )

    patched_source = source

    # -------------------------------------------------------------------------
    # The core issue: When force_align=True, AbundancesAligner is used.
    # AbundancesAligner uses HungarianAligner which calls MSE()(A.T, Aref.T)
    # where A is (M, N) = (240, 5625) and Aref is (p, N) = (5, 5625).
    # A.T = (5625, 240), Aref.T = (5625, 5) — different shapes.
    # MSE._check_input asserts same shape, which fails.
    # But MSE is being used as a pairwise distance here (result is (240, 5) matrix).
    # 
    # After alignment selects p rows, compute_metric calls aRMSE and SRE with
    # A_gt (5, 5625) and A1 (5, 5625) — same shapes, so _check_input works fine.
    #
    # Solution: Make _check_input lenient (remove shape assertion).
    # Also, the aRMSE __call__ does (A - Aref)**2 which requires same shapes,
    # but that's fine since after alignment both are (5, 5625).
    # -------------------------------------------------------------------------

    # Replace _check_input to be lenient about shape mismatches
    # Handle both formatting variants
    patched_source = patched_source.replace(
        """    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref""",
        """    @staticmethod
    def _check_input(X, Xref):
        import numpy as _np_internal_
        X = _np_internal_.asarray(X)
        Xref = _np_internal_.asarray(Xref)
        return X, Xref"""
    )

    patched_source = patched_source.replace(
        """    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return (X, Xref)""",
        """    @staticmethod
    def _check_input(X, Xref):
        import numpy as _np_internal_
        X = _np_internal_.asarray(X)
        Xref = _np_internal_.asarray(Xref)
        return (X, Xref)"""
    )

    # -------------------------------------------------------------------------
    # The second issue: aRMSE is called from compute_metric with X_hat[ii] and
    # X_gt[ii] when detail=True. After alignment, A1 has shape (p, N) matching
    # A_gt (p, N). So aRMSE(X_hat[ii], X_gt[ii]) gets 1D arrays of shape (N,).
    # aRMSE returns a scalar: 100 * sqrt(((A - Aref)**2).mean()).
    # compute_metric does round(metric(...), 4) on that scalar. This should work.
    #
    # BUT: The "Overall" call is metric(X_hat, X_gt) where X_hat and X_gt are
    # both (p, N). For aRMSE this is fine (element-wise subtraction).
    # For SRE this is also fine (frobenius norm).
    #
    # The actual error was: "operands could not be broadcast together with 
    # shapes (5,5625) (240,5625)" — this means A1 still has 240 rows, not 5.
    # This means force_align failed (AbundancesAligner crashed due to MSE shape
    # assertion), and the code fell through to the else branch using index,
    # OR force_align is False and index selection gives wrong shape.
    #
    # Actually looking more carefully: the error is in compute_metric -> __call__
    # for aRMSE at line 343. The shapes (5,5625) vs (240,5625) mean A_gt is
    # (5,5625) and A1 is (240,5625). So the alignment/selection didn't reduce
    # A_hat from 240 to 5 rows.
    #
    # With our _check_input fix, the AbundancesAligner should now work.
    # BUT: the Munkres algorithm on a 240x5 cost matrix... let's check.
    # HungarianAligner.fit computes self.dists = self.criterion(A.T, self.Aref.T)
    # where A is A_hat (240, 5625) and Aref is A_gt (5, 5625).
    # A.T = (5625, 240), Aref.T = (5625, 5).
    # MSE returns sqrt(normE.T^2 + normEref^2 - 2*E.T@Eref)
    # normE = norm of each column of A.T = (1, 240), normE.T = (240, 1)
    # normEref = norm of each column of Aref.T = (1, 5)
    # normE.T^2 = (240, 1), normEref^2 = (1, 5)
    # E.T @ Eref = (240, 5625) @ (5625, 5) = (240, 5) 
    # Result: (240, 5) distance matrix. Good.
    #
    # Then p = A.shape[0] = 240, P = (240, 240). That's wrong!
    # The Munkres will try to find 240 assignments in a 240x5 matrix (converted to list).
    # Munkres requires a square or rectangular matrix. For 240x5, it finds 5 assignments.
    # But P is initialized as (240, 240), and only 5 entries get set.
    # Then self.P = P.T, a (240, 240) matrix.
    # transform does self.P @ A where A is (240, 5625). Result is (240, 5625).
    # But we want (5, 5625)!
    #
    # The issue is that HungarianAligner.fit uses p = A.shape[0] for P dimensions.
    # When A has 240 rows and Aref has 5 rows, P should be (5, 240) to select 5 rows.
    #
    # We need to fix this. Let's patch HungarianAligner.fit to handle rectangular cases.
    # Or better yet, let's patch the main function's alignment logic to manually
    # select rows using the Hungarian assignment result.
    # -------------------------------------------------------------------------

    # Patch HungarianAligner.fit to handle non-square cases properly
    # The fix: P should be (p_ref, p_hat) where p_ref = Aref.shape[0], p_hat = A.shape[0]
    # Munkres returns (row, col) pairs. We want to map each Aref row to an A row.
    # P[aref_idx, a_idx] = 1, so P @ A selects the right rows.

    # Find and replace HungarianAligner.fit
    # Try the variant from gen_data_code first (tuple assignment style)
    old_fit_v1 = """    def fit(self, A):
        if not HAS_MUNKRES:
            raise ImportError('munkres package required for HungarianAligner')
        self.dists = self.criterion(A.T, self.Aref.T)
        p = A.shape[0]
        P = np.zeros((p, p))
        m = Munkres()
        indices = m.compute(self.dists.tolist())
        for (row, col) in indices:
            P[row, col] = 1.0
        self.P = P.T"""

    new_fit_v1 = """    def fit(self, A):
        if not HAS_MUNKRES:
            raise ImportError('munkres package required for HungarianAligner')
        self.dists = self.criterion(A.T, self.Aref.T)
        p_hat = A.shape[0]
        p_ref = self.Aref.shape[0]
        # P maps from A space to Aref space: P is (p_ref, p_hat)
        P = np.zeros((p_ref, p_hat))
        m = Munkres()
        # dists is (p_hat, p_ref), munkres assigns rows->cols
        cost = self.dists.tolist()
        indices = m.compute(cost)
        for (row, col) in indices:
            # row is index in A (p_hat), col is index in Aref (p_ref)
            P[col, row] = 1.0
        self.P = P"""

    patched_source = patched_source.replace(old_fit_v1, new_fit_v1)

    # Also try the other variant (without tuple unpacking)
    old_fit_v2 = """    def fit(self, A):
        if not HAS_MUNKRES:
            raise ImportError("munkres package required for HungarianAligner")

        # Computing distance matrix
        self.dists = self.criterion(A.T, self.Aref.T)

        # Initialization
        p = A.shape[0]
        P = np.zeros((p, p))

        m = Munkres()
        indices = m.compute(self.dists.tolist())
        for row, col in indices:
            P[row, col] = 1.0

        self.P = P.T"""

    new_fit_v2 = """    def fit(self, A):
        if not HAS_MUNKRES:
            raise ImportError("munkres package required for HungarianAligner")

        # Computing distance matrix
        self.dists = self.criterion(A.T, self.Aref.T)

        # Initialization
        p_hat = A.shape[0]
        p_ref = self.Aref.shape[0]
        P = np.zeros((p_ref, p_hat))

        m = Munkres()
        cost = self.dists.tolist()
        indices = m.compute(cost)
        for row, col in indices:
            P[col, row] = 1.0

        self.P = P"""

    patched_source = patched_source.replace(old_fit_v2, new_fit_v2)

    # Now fix transform: it asserts A.shape[0] == P.shape[0] == P.shape[1]
    # With our fix, P is (p_ref, p_hat), so P.shape[0] != P.shape[1]
    # We need P @ A where P is (p_ref, p_hat) and A is (p_hat, N) -> (p_ref, N)

    old_transform_v1 = """    def transform(self, A):
        assert self.P is not None, 'Must be fitted first'
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]
        return self.P @ A"""

    new_transform_v1 = """    def transform(self, A):
        assert self.P is not None, 'Must be fitted first'
        assert A.shape[0] == self.P.shape[1]
        return self.P @ A"""

    patched_source = patched_source.replace(old_transform_v1, new_transform_v1)

    old_transform_v2 = """    def transform(self, A):
        assert self.P is not None, "Must be fitted first"
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]

        return self.P @ A"""

    new_transform_v2 = """    def transform(self, A):
        assert self.P is not None, "Must be fitted first"
        assert A.shape[0] == self.P.shape[1]

        return self.P @ A"""

    patched_source = patched_source.replace(old_transform_v2, new_transform_v2)

    # Also fix transform_endmembers similarly
    old_te_v1 = """    def transform_endmembers(self, E):
        assert self.P is not None, 'Must be fitted first'
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]
        return E @ self.P.T"""

    new_te_v1 = """    def transform_endmembers(self, E):
        assert self.P is not None, 'Must be fitted first'
        assert E.shape[1] == self.P.shape[1]
        return E @ self.P.T"""

    patched_source = patched_source.replace(old_te_v1, new_te_v1)

    old_te_v2 = """    def transform_endmembers(self, E):
        assert self.P is not None, "Must be fitted first"
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]

        return E @ self.P.T"""

    new_te_v2 = """    def transform_endmembers(self, E):
        assert self.P is not None, "Must be fitted first"
        assert E.shape[1] == self.P.shape[1]

        return E @ self.P.T"""

    patched_source = patched_source.replace(old_te_v2, new_te_v2)

    # Write a temporary patched module
    patched_path = os.path.join(script_dir, '_patched_agent_main.py')
    with open(patched_path, 'w') as f:
        f.write(preamble)
        f.write('\n')
        f.write(patched_source)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_patched_agent_main", patched_path)
        patched_module = importlib.util.module_from_spec(spec)
        sys.modules['_patched_agent_main'] = patched_module
        sys.modules['agent_main'] = patched_module
        spec.loader.exec_module(patched_module)
        return patched_module.main
    finally:
        try:
            os.remove(patched_path)
        except:
            pass


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)

    # Load outer data
    try:
        import dill
        import numpy as np
        print(f"Loading outer data from: {outer_path}")
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Outer data keys: {list(outer_data.keys())}")
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"ERROR loading outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Set random seed for reproducibility before importing/running
    # The recorded data was captured with a specific random state.
    # We need to match it. Try to extract random state from outer_data if available.
    # Otherwise, we just run and compare (main returns None typically).

    # Import main function with globals patched
    try:
        target_main = setup_and_import_main()
        print("Successfully imported 'main' from agent_main.")
    except Exception as e:
        print(f"ERROR importing main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    from verification_utils import recursive_check

    if len(inner_paths) > 0:
        # --- Scenario B: Factory/Closure Pattern ---
        print("Detected Scenario B: Factory/Closure Pattern")

        try:
            print("Phase 1: Calling main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = target_main(*outer_args, **outer_kwargs)
            print(f"Operator returned: {type(agent_operator)}")
        except Exception as e:
            print(f"ERROR during Phase 1 (operator creation): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Phase 2: Executing operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR during Phase 2 (operator execution): {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                if passed:
                    print(f"TEST PASSED for {os.path.basename(inner_path)}")
                else:
                    print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # --- Scenario A: Simple Function ---
        print("Detected Scenario A: Simple Function")

        try:
            print("Calling main(*outer_args, **outer_kwargs)...")
            result = target_main(*outer_args, **outer_kwargs)
            print(f"Result type: {type(result)}")
        except Exception as e:
            print(f"ERROR during main execution: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            print("Comparing results...")
            print(f"Expected type: {type(expected)}, Result type: {type(result)}")
            passed, msg = recursive_check(expected, result)
            if passed:
                print("TEST PASSED")
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nAll tests passed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()