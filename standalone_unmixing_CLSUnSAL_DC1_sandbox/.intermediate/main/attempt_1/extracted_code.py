import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-define SCRIPT_DIR and other missing globals before importing agent_main
import agent_main as _agent_module_preload
# We need to patch the module's globals before it fully initializes.
# Since the error occurs at module load time, we need to set SCRIPT_DIR before import.
# Let's do it differently:

def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) paths
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

    # Import main function - need to handle missing globals
    try:
        # Determine the script directory for agent_main
        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        agent_dir = os.path.dirname(os.path.abspath(agent_main_path))

        # Read agent_main.py source and patch missing variables
        with open(agent_main_path, 'r') as f:
            source = f.read()

        # We need to inject SCRIPT_DIR, CONFIG, logging, and HAS_MUNKRES/Munkres before the module loads
        import types
        import importlib

        # Create a patched version by prepending necessary definitions
        preamble = f"""
import os
import logging
import sys

SCRIPT_DIR = {repr(agent_dir)}

# Try to load config
_config_path = os.path.join(SCRIPT_DIR, 'config.json')
_config_path2 = os.path.join(SCRIPT_DIR, 'run_code', 'config.json')
CONFIG = {{}}
import json
for _cp in [_config_path, _config_path2]:
    if os.path.exists(_cp):
        with open(_cp, 'r') as _f:
            CONFIG = json.load(_f)
        break

if not CONFIG:
    # Try to find any config file
    for root, dirs, files in os.walk(SCRIPT_DIR):
        for fn in files:
            if fn == 'config.json':
                with open(os.path.join(root, fn), 'r') as _f:
                    CONFIG = json.load(_f)
                break
        if CONFIG:
            break

# Set defaults if not found
if 'EPS' not in CONFIG:
    CONFIG['EPS'] = 1e-10

logging.basicConfig(level=logging.INFO)

try:
    from munkres import Munkres
    HAS_MUNKRES = True
except ImportError:
    HAS_MUNKRES = False
    class Munkres:
        pass

"""
        # Write a temporary patched module
        patched_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_patched_agent_main.py')
        with open(patched_path, 'w') as f:
            f.write(preamble)
            f.write('\n')
            f.write(source)

        # Import the patched module
        spec = importlib.util.spec_from_file_location("_patched_agent_main", patched_path)
        patched_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patched_module)

        target_main = patched_module.main
        print("Successfully imported 'main' from patched agent_main.")

        # Clean up
        try:
            os.remove(patched_path)
        except:
            pass

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
                print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            except Exception as e:
                print(f"ERROR loading inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                print("Phase 2: Executing operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
                print(f"Result type: {type(result)}")
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


import sys
import os
import dill
import numpy as np
import traceback
import importlib
import json
import logging

# Ensure the current directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Before importing agent_main, we need to set up globals it expects
# We'll do this by creating a temporary wrapper module approach
# First, find and load the config
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


def setup_and_import_main():
    """Import main from agent_main with proper globals setup."""
    agent_main_path = os.path.join(script_dir, 'agent_main.py')

    # Read the source
    with open(agent_main_path, 'r') as f:
        source = f.read()

    # Find config
    config = find_config(script_dir)
    if 'EPS' not in config:
        config['EPS'] = 1e-10

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Try to import munkres
    try:
        from munkres import Munkres
        has_munkres = True
    except ImportError:
        has_munkres = False

    # Create module namespace with required globals
    module = types.ModuleType('agent_main')
    module.__file__ = agent_main_path
    module.__name__ = 'agent_main'

    # Inject the missing globals into the module's namespace before exec
    module_globals = module.__dict__
    module_globals['SCRIPT_DIR'] = script_dir
    module_globals['CONFIG'] = config
    module_globals['HAS_MUNKRES'] = has_munkres
    if has_munkres:
        module_globals['Munkres'] = Munkres

    # Register in sys.modules so internal imports work
    sys.modules['agent_main'] = module

    # Execute the module code
    code = compile(source, agent_main_path, 'exec')
    exec(code, module_globals)

    return module_globals['main']


import types


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