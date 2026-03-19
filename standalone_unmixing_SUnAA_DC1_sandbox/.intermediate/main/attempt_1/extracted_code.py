import sys
import os
import dill
import numpy as np
import traceback
import logging

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch the agent_main module before importing it
# We need to ensure 'logging' and 'CONFIG' are available in agent_main's namespace
import agent_main

# Check if logging is missing and inject it
if not hasattr(agent_main, 'logging'):
    agent_main.logging = logging

# Check if CONFIG is missing and try to load it
if not hasattr(agent_main, 'CONFIG') or 'CONFIG' not in dir(agent_main):
    # Try to find and load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            agent_main.CONFIG = json.load(f)
    else:
        # Try yaml
        config_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        if os.path.exists(config_yaml):
            import yaml
            with open(config_yaml, 'r') as f:
                agent_main.CONFIG = yaml.safe_load(f)

# Now reimport to get the properly initialized module
# Force reload
import importlib
try:
    importlib.reload(agent_main)
except Exception:
    pass

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