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

    # We need to inject globals BEFORE agent_main.py executes at module level.
    # We'll use builtins or compile+exec approach with pre-set globals.

    # Build the preamble that defines missing globals
    preamble = """
import os as os
import sys as sys
import logging as logging
import json as _json_

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

    # Write a temporary patched module
    patched_path = os.path.join(script_dir, '_patched_agent_main.py')
    with open(patched_path, 'w') as f:
        f.write(preamble)
        f.write('\n')
        f.write(source)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_patched_agent_main", patched_path)
        patched_module = importlib.util.module_from_spec(spec)
        sys.modules['_patched_agent_main'] = patched_module
        # Also register as agent_main so internal references work
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