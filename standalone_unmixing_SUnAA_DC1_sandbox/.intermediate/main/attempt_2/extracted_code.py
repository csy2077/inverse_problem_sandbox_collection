import sys
import os
import dill
import numpy as np
import traceback
import logging

# We need to patch agent_main before it's imported.
# First, read the agent_main source and figure out what globals it needs.
# The module uses 'logging' and 'CONFIG' at module level.

# Inject logging into the agent_main module's namespace before import
agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')

# Try to find CONFIG first
config = None
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
if os.path.exists(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if os.path.exists(config_yaml):
        import yaml
        with open(config_yaml, 'r') as f:
            config = yaml.safe_load(f)

if config is None:
    # Try to find config.py
    config_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    if os.path.exists(config_py):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib
        config_mod = importlib.import_module('config')
        if hasattr(config_mod, 'CONFIG'):
            config = config_mod.CONFIG

if config is None:
    # Provide a default CONFIG that matches the expected structure
    config = {
        "seed": 0,
        "SNR": 30,
        "dataset": "DC1",
        "data_dir": "./data",
        "figs_dir": "./figs",
        "l2_normalization": False,
        "projection": False,
        "force_align": False,
        "EPS": 1e-9,
        "model": {
            "T": 100,
            "low_rank": False,
        }
    }

# Now we need to create the agent_main module manually with the right globals
import types
import importlib

# Create a fresh module
agent_main_module = types.ModuleType('agent_main')
agent_main_module.__file__ = agent_main_path
agent_main_module.__name__ = 'agent_main'

# Set up the namespace with required globals
agent_main_module.logging = logging
agent_main_module.CONFIG = config

# Register it in sys.modules BEFORE executing it
sys.modules['agent_main'] = agent_main_module

# Read and execute the source
with open(agent_main_path, 'r') as f:
    source = f.read()

# Compile and exec in the module's namespace
try:
    code = compile(source, agent_main_path, 'exec')
    exec(code, agent_main_module.__dict__)
except Exception as e:
    print(f"Error loading agent_main: {e}")
    traceback.print_exc()
    sys.exit(1)

from agent_main import main
from verification_utils import recursive_check


def load_pkl(path):
    """Load a pickle file using dill."""
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def main_test():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Separate outer (main) data from inner (parent_function) data
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, f"Could not find outer data file in {data_paths}"

    # Phase 1: Load outer data
    try:
        print(f"Loading outer data from: {outer_path}")
        outer_data = load_pkl(outer_path)
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        outer_output = outer_data.get('output', None)
        print(f"Outer data loaded successfully. func_name={outer_data.get('func_name', 'N/A')}")
        print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute main
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        try:
            print("Calling main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Operator obtained: {type(agent_operator)}")
            assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        except Exception as e:
            print(f"FAILED during main() call (Phase 1): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Process each inner data file
        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                inner_data = load_pkl(inner_path)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
                print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"FAILED to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Executing operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution completed. Result type: {type(result)}")
            except Exception as e:
                print(f"FAILED during operator execution: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Comparing results...")
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print(f"Inner test passed for {os.path.basename(inner_path)}")
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        try:
            print("Calling main(*args, **kwargs)...")
            result = main(*outer_args, **outer_kwargs)
            print(f"Execution completed. Result type: {type(result)}")
        except Exception as e:
            print(f"FAILED during main() call: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            print("Comparing results...")
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"FAILED during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main_test()