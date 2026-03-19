import sys
import os
import dill
import numpy as np
import traceback
import logging
import types
import importlib
import importlib.util
import json
import unittest.mock as mock

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

script_dir = os.path.dirname(os.path.abspath(__file__))
agent_main_path = os.path.join(script_dir, 'agent_main.py')

# Read the agent_main source
with open(agent_main_path, 'r') as f:
    agent_main_source = f.read()

# Try to find CONFIG
CONFIG = None

# Try config.py
config_py_path = os.path.join(script_dir, 'config.py')
if os.path.exists(config_py_path):
    try:
        spec = importlib.util.spec_from_file_location("config_module", config_py_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        if hasattr(config_module, 'CONFIG'):
            CONFIG = config_module.CONFIG
    except Exception as e:
        print(f"Warning: could not load config.py: {e}")

# Try config.json
if CONFIG is None:
    config_path = os.path.join(script_dir, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                CONFIG = json.load(f)
        except Exception as e:
            print(f"Warning: could not load config.json: {e}")

# Try config.yaml
if CONFIG is None:
    config_yaml_path = os.path.join(script_dir, 'config.yaml')
    if os.path.exists(config_yaml_path):
        try:
            import yaml
            with open(config_yaml_path, 'r') as f:
                CONFIG = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: could not load config.yaml: {e}")

# Search for any config-like file
if CONFIG is None:
    for fname in sorted(os.listdir(script_dir)):
        if 'config' in fname.lower():
            full = os.path.join(script_dir, fname)
            if fname.endswith('.json'):
                try:
                    with open(full, 'r') as f:
                        CONFIG = json.load(f)
                    if CONFIG is not None:
                        break
                except:
                    pass
            elif fname.endswith(('.yaml', '.yml')):
                try:
                    import yaml
                    with open(full, 'r') as f:
                        CONFIG = yaml.safe_load(f)
                    if CONFIG is not None:
                        break
                except:
                    pass
            elif fname.endswith('.py') and fname != 'config.py':
                try:
                    spec = importlib.util.spec_from_file_location("config_module2", full)
                    cm = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(cm)
                    if hasattr(cm, 'CONFIG'):
                        CONFIG = cm.CONFIG
                        break
                except:
                    pass

# Fallback default CONFIG
if CONFIG is None:
    CONFIG = {
        "seed": 0,
        "SNR": 30,
        "dataset": "DC1",
        "data_dir": "./data",
        "figs_dir": "./figs",
        "l2_normalization": False,
        "projection": False,
        "force_align": False,
        "EPS": 1e-10,
        "model": {
            "AL_iters": 5,
            "lambd": 0.0,
            "verbose": True,
            "tol": 1e-4,
            "x0": 0,
        }
    }

# Fix data_dir to use absolute path based on script_dir
# Look for the actual data directory
possible_data_dirs = [
    os.path.join(script_dir, 'data'),
    os.path.join(script_dir, 'run_code', 'data'),
    os.path.join(os.path.dirname(script_dir), 'data'),
    os.path.join(os.path.dirname(script_dir), 'run_code', 'data'),
]

# Also search for the .mat file
dataset_name = CONFIG.get("dataset", "DC1")
mat_filename = f"{dataset_name}.mat"

found_data_dir = None
for dd in possible_data_dirs:
    mat_path = os.path.join(dd, mat_filename)
    if os.path.exists(mat_path):
        found_data_dir = dd
        break

# Broader search if not found
if found_data_dir is None:
    for root, dirs, files in os.walk(script_dir):
        if mat_filename in files:
            found_data_dir = root
            break

if found_data_dir is None:
    # Search parent directories
    parent = os.path.dirname(script_dir)
    for root, dirs, files in os.walk(parent):
        if mat_filename in files:
            found_data_dir = root
            break

if found_data_dir is None:
    # Even broader: search common locations
    search_base = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_S2WSU_DC1_sandbox'
    for root, dirs, files in os.walk(search_base):
        if mat_filename in files:
            found_data_dir = root
            break

if found_data_dir is not None:
    CONFIG["data_dir"] = found_data_dir
    print(f"Found data directory: {found_data_dir}")
else:
    print(f"WARNING: Could not find {mat_filename} anywhere. Test may fail.")

# Fix figs_dir to use absolute path and ensure it exists
figs_dir = CONFIG.get("figs_dir", "./figs")
if not os.path.isabs(figs_dir):
    figs_dir = os.path.join(script_dir, figs_dir)
CONFIG["figs_dir"] = figs_dir
os.makedirs(figs_dir, exist_ok=True)

from verification_utils import recursive_check


def load_pkl(path):
    """Load a pickle file using dill."""
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def test_main():
    """Test the main function against recorded standard data."""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_S2WSU_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Separate outer (main) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    assert outer_path is not None, f"Could not find outer data file (data_main.pkl) in {data_paths}"

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

    # Now import agent_main with the corrected CONFIG
    # We need to patch CONFIG before importing
    try:
        # Create a modified version of agent_main with correct paths
        agent_main_mod = types.ModuleType('agent_main')
        agent_main_mod.__file__ = agent_main_path
        agent_main_mod.__name__ = 'agent_main'
        agent_main_mod.__dict__['logging'] = logging
        agent_main_mod.__dict__['CONFIG'] = CONFIG

        # Check if source needs CONFIG injection
        needs_config = True
        for line in agent_main_source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('CONFIG') and '=' in stripped and not stripped.startswith('CONFIG[') and not stripped.startswith('CONFIG.'):
                needs_config = False
                break
        if needs_config:
            if 'CONFIG' in agent_main_source and ('json.load' in agent_main_source or 'yaml.safe_load' in agent_main_source or 'load_config' in agent_main_source):
                needs_config = False

        # We need to handle the SCRIPT_DIR issue
        # The agent_main.py uses SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        # and then rel_path uses SCRIPT_DIR. We need data_dir to be absolute.

        # Modify the source to inject CONFIG with absolute paths
        patched_source = "import logging\n"

        if needs_config:
            patched_source += f"CONFIG = {json.dumps(CONFIG)}\n"
        else:
            # Replace CONFIG loading with our CONFIG
            # We'll inject it before the source and override
            patched_source += f"CONFIG = {json.dumps(CONFIG)}\n"
            # Remove the original CONFIG loading lines
            lines = agent_main_source.split('\n')
            new_lines = []
            skip_block = False
            for line in lines:
                stripped = line.strip()
                # Skip lines that define/load CONFIG
                if stripped.startswith('CONFIG') and '=' in stripped and not stripped.startswith('CONFIG[') and not stripped.startswith('CONFIG.'):
                    skip_block = True
                    continue
                if skip_block:
                    # Check if next meaningful line is part of config loading
                    if stripped and not stripped.startswith('#'):
                        skip_block = False
                new_lines.append(line)
            agent_main_source_modified = '\n'.join(new_lines)
            patched_source += agent_main_source_modified

        if needs_config:
            patched_source += agent_main_source

        sys.modules['agent_main'] = agent_main_mod

        # Also handle the rel_path function - if data_dir is absolute, it should work
        # The issue is that rel_path joins SCRIPT_DIR with data_dir
        # If data_dir is absolute, os.path.join(SCRIPT_DIR, absolute_path) = absolute_path on Unix
        # So making data_dir absolute should fix the issue

        code = compile(patched_source, agent_main_path, 'exec')
        exec(code, agent_main_mod.__dict__)

        main = agent_main_mod.main
        print("agent_main loaded successfully with corrected CONFIG")
    except Exception as e:
        print(f"Failed to load agent_main: {e}")
        traceback.print_exc()

        # Try alternative: directly modify and re-exec
        try:
            print("Trying alternative loading approach...")
            alt_source = "import logging\n"
            alt_source += f"CONFIG = {json.dumps(CONFIG)}\n"

            # Strip any existing CONFIG definition from source
            filtered_lines = []
            in_config_block = False
            paren_depth = 0
            for line in agent_main_source.split('\n'):
                stripped = line.strip()
                if not in_config_block:
                    if (stripped.startswith('CONFIG') and '=' in stripped
                            and not stripped.startswith('CONFIG[')
                            and not stripped.startswith('CONFIG.')):
                        in_config_block = True
                        paren_depth = stripped.count('{') - stripped.count('}')
                        paren_depth += stripped.count('(') - stripped.count(')')
                        if paren_depth <= 0:
                            in_config_block = False
                        continue
                    filtered_lines.append(line)
                else:
                    paren_depth += stripped.count('{') - stripped.count('}')
                    paren_depth += stripped.count('(') - stripped.count(')')
                    if paren_depth <= 0:
                        in_config_block = False
                    continue

            alt_source += '\n'.join(filtered_lines)

            agent_main_mod2 = types.ModuleType('agent_main')
            agent_main_mod2.__file__ = agent_main_path
            agent_main_mod2.__dict__['logging'] = logging
            agent_main_mod2.__dict__['CONFIG'] = CONFIG
            sys.modules['agent_main'] = agent_main_mod2

            code = compile(alt_source, agent_main_path, 'exec')
            exec(code, agent_main_mod2.__dict__)
            agent_main_mod = agent_main_mod2
            main = agent_main_mod.main
            print("Alternative loading succeeded")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            traceback.print_exc()
            sys.exit(1)

    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        try:
            print("Running main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"Operator obtained: {type(agent_operator)}")
            assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        except Exception as e:
            print(f"FAILED to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                print(f"Loading inner data from: {inner_path}")
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
                print("Executing operator with inner args...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Execution complete. Result type: {type(result)}")
            except Exception as e:
                print(f"FAILED to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"TEST FAILED: {msg}")
                    sys.exit(1)
                else:
                    print("TEST PASSED")
                    sys.exit(0)
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        try:
            print("Running main(*args, **kwargs)...")
            result = main(*outer_args, **outer_kwargs)
            print(f"Execution complete. Result type: {type(result)}")
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print("TEST PASSED")
                sys.exit(0)
        except Exception as e:
            print(f"FAILED during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    test_main()