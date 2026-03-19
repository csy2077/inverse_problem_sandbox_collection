import sys
import os
import traceback

# Pre-inject logging into the agent_main module before it fully loads
import logging

# We need to patch the agent_main module's globals before importing it.
# First, read and identify what's needed.
agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')

# Patch by injecting logging and other missing names into the module's namespace
# We'll do this by modifying sys.modules and using importlib
import importlib
import types
from pathlib import Path

# Create a temporary module to pre-populate
_agent_module = types.ModuleType('agent_main')
_agent_module.__file__ = agent_main_path
_agent_module.__name__ = 'agent_main'
_agent_module.logging = logging
_agent_module.Path = Path

# We need to figure out SCRIPT_DIR and CONFIG before the module loads
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(agent_main_path)))
_agent_module.SCRIPT_DIR = SCRIPT_DIR

# Try to find and load CONFIG - check for config files
config_path_candidates = [
    SCRIPT_DIR / 'config.json',
    SCRIPT_DIR / 'config.yaml',
    SCRIPT_DIR / 'config.yml',
    SCRIPT_DIR / 'config.toml',
    SCRIPT_DIR / 'standalone_config.json',
]

CONFIG = {
    'EPS': 1e-10,
    'seed': 0,
    'SNR': None,
    'l2_normalization': False,
    'projection': False,
    'data': {'dataset': 'DC1'},
    'EDAA': {
        'T': 100,
        'K1': 5,
        'K2': 5,
        'M': 50,
        'normalize': True,
    }
}

import json as _json

for cp in config_path_candidates:
    if os.path.isfile(cp):
        try:
            with open(cp, 'r') as _f:
                loaded_config = _json.load(_f)
                CONFIG.update(loaded_config)
                print(f"[INFO] Loaded config from {cp}")
                break
        except Exception:
            pass

# Also check for a Python config file
config_py = SCRIPT_DIR / 'config.py'
if os.path.isfile(config_py):
    try:
        spec = importlib.util.spec_from_file_location("_config_module", config_py)
        config_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_mod)
        if hasattr(config_mod, 'CONFIG'):
            CONFIG.update(config_mod.CONFIG)
            print(f"[INFO] Loaded CONFIG from config.py")
    except Exception:
        pass

_agent_module.CONFIG = CONFIG

# Now actually load the agent_main source with these pre-injected globals
try:
    with open(agent_main_path, 'r') as f:
        source = f.read()

    # Compile and exec with the pre-populated namespace
    code = compile(source, agent_main_path, 'exec')
    exec(code, _agent_module.__dict__)
    sys.modules['agent_main'] = _agent_module
    print("[INFO] agent_main loaded successfully with patched globals")
except Exception as e:
    print(f"[FAIL] Failed to load agent_main: {e}")
    traceback.print_exc()
    sys.exit(1)

import dill
import torch
import numpy as np

from agent_main import main
from verification_utils import recursive_check


def main_test():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_EDAA_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, f"No outer data file found in {data_paths}"
    print(f"[INFO] Outer data path: {outer_path}")
    print(f"[INFO] Inner data paths: {inner_paths}")

    # Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"[INFO] Outer data loaded successfully. Keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"[FAIL] Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"[INFO] outer_args type: {type(outer_args)}, length: {len(outer_args) if isinstance(outer_args, (list, tuple)) else 'N/A'}")
    print(f"[INFO] outer_kwargs type: {type(outer_kwargs)}, keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")
    print(f"[INFO] expected_output type: {type(expected_output)}")

    if len(inner_paths) > 0:
        # --- Scenario B: Factory/Closure Pattern ---
        print("[INFO] Detected Scenario B: Factory/Closure Pattern")

        # Phase 1: Create operator
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"[INFO] Operator created successfully. Type: {type(agent_operator)}")
        except Exception as e:
            print(f"[FAIL] Failed to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"

        # Phase 2: Execute with inner data
        for inner_path in inner_paths:
            print(f"[INFO] Loading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"[INFO] Inner data loaded. Keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"[FAIL] Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"[INFO] Operator executed successfully. Result type: {type(result)}")
            except Exception as e:
                print(f"[FAIL] Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(inner_expected, result)
                if passed:
                    print(f"[PASS] Inner data check passed for {os.path.basename(inner_path)}")
                else:
                    print(f"[FAIL] Inner data check failed for {os.path.basename(inner_path)}: {msg}")
                    sys.exit(1)
            except Exception as e:
                print(f"[FAIL] Verification error: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # --- Scenario A: Simple Function ---
        print("[INFO] Detected Scenario A: Simple Function")

        # Phase 1: Run function
        try:
            result = main(*outer_args, **outer_kwargs)
            print(f"[INFO] Function executed successfully. Result type: {type(result)}")
        except Exception as e:
            print(f"[FAIL] Failed to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Phase 2: Compare
        try:
            passed, msg = recursive_check(expected_output, result)
            if passed:
                print("[PASS] Output check passed.")
            else:
                print(f"[FAIL] Output check failed: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"[FAIL] Verification error: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main_test()