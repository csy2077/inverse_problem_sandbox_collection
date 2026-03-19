import sys
import os
import dill
import numpy as np
import traceback
import logging

# We need to find the data directory first
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_CODE_DIR = os.path.join(SCRIPT_DIR, 'run_code')

# Search for the actual data file DC1.mat
def find_data_file(filename='DC1.mat', search_roots=None):
    if search_roots is None:
        search_roots = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                return dirpath
    return None

data_dir_abs = find_data_file('DC1.mat')

# Try to load config
config = None
for candidate in [
    os.path.join(SCRIPT_DIR, 'config.json'),
    os.path.join(RUN_CODE_DIR, 'config.json'),
]:
    if os.path.exists(candidate):
        import json
        with open(candidate, 'r') as f:
            config = json.load(f)
        break

if config is None:
    for candidate in [
        os.path.join(SCRIPT_DIR, 'config.yaml'),
        os.path.join(RUN_CODE_DIR, 'config.yaml'),
    ]:
        if os.path.exists(candidate):
            import yaml
            with open(candidate, 'r') as f:
                config = yaml.safe_load(f)
            break

if config is None:
    for candidate in [
        os.path.join(SCRIPT_DIR, 'config.py'),
        os.path.join(RUN_CODE_DIR, 'config.py'),
    ]:
        if os.path.exists(candidate):
            sys.path.insert(0, os.path.dirname(candidate))
            import importlib
            config_mod = importlib.import_module('config')
            if hasattr(config_mod, 'CONFIG'):
                config = config_mod.CONFIG
            break

if config is None:
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

# If we found the data file, update config's data_dir to be relative to SCRIPT_DIR
# or we can patch the rel_path / SCRIPT_DIR in agent_main
if data_dir_abs is not None:
    config["data_dir"] = data_dir_abs

# Also ensure figs_dir is absolute and exists
figs_dir = config.get("figs_dir", "./figs")
if not os.path.isabs(figs_dir):
    figs_dir = os.path.join(SCRIPT_DIR, figs_dir)
config["figs_dir"] = figs_dir
os.makedirs(figs_dir, exist_ok=True)

# Now create the agent_main module manually with the right globals
import types

agent_main_path = os.path.join(SCRIPT_DIR, 'agent_main.py')
if not os.path.exists(agent_main_path):
    agent_main_path = os.path.join(RUN_CODE_DIR, 'agent_main.py')

agent_main_module = types.ModuleType('agent_main')
agent_main_module.__file__ = agent_main_path
agent_main_module.__name__ = 'agent_main'

agent_main_module.logging = logging
agent_main_module.CONFIG = config

sys.modules['agent_main'] = agent_main_module

with open(agent_main_path, 'r') as f:
    source = f.read()

# We need to patch SCRIPT_DIR in agent_main so that rel_path works correctly
# when data_dir is already absolute
# We'll patch it by overriding after exec

try:
    code = compile(source, agent_main_path, 'exec')
    exec(code, agent_main_module.__dict__)
except Exception as e:
    print(f"Error loading agent_main: {e}")
    traceback.print_exc()
    sys.exit(1)

# Patch SCRIPT_DIR so rel_path resolves correctly
# If data_dir is absolute, rel_path(data_dir, ...) = os.path.join(SCRIPT_DIR, abs_path, ...) which is wrong
# We need rel_path to handle absolute paths properly
# Override rel_path in the module
original_rel_path = agent_main_module.rel_path
def patched_rel_path(*args):
    if args and os.path.isabs(args[0]):
        return os.path.join(*args)
    return original_rel_path(*args)

agent_main_module.rel_path = patched_rel_path
# Also update SCRIPT_DIR in the module
agent_main_module.SCRIPT_DIR = SCRIPT_DIR

from agent_main import main
from verification_utils import recursive_check


def load_pkl(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def main_test():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

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

        for inner_path in inner_paths:
            try:
                print(f"\nLoading inner data from: {inner_path}")
                inner_data = load_pkl(inner_path)
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                expected = inner_data.get('output', None)
                print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
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