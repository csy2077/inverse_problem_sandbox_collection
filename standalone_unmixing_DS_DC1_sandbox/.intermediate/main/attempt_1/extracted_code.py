import sys
import os
import logging
import dill
import numpy as np
import traceback

# Ensure logging is available before importing agent_main
import logging

# Patch agent_main's module-level dependencies before importing
try:
    import agent_main
except Exception:
    pass

# We need to ensure the agent_main module can be imported properly
# The issue is that agent_main.py uses 'logging' and 'CONFIG' without importing/defining them
# We need to patch these before the import

# First, let's try to set up the module environment
try:
    # Create a minimal config if needed
    import importlib
    # Check if agent_main needs patching
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    
    # Read the source to understand what's needed
    with open(agent_main_path, 'r') as f:
        source = f.read()
    
    # Check if CONFIG and logging are missing
    needs_logging = 'logging' in source and 'import logging' not in source
    needs_config = 'CONFIG' in source and 'CONFIG' not in source.split('CONFIG')[0].split('\n')[-1]
    
except Exception:
    pass

# Try to handle the import by injecting required globals
try:
    import types
    
    # If agent_main is not yet properly loaded, we need to handle it
    if 'agent_main' not in sys.modules or not hasattr(sys.modules.get('agent_main', None), 'main'):
        # Remove partial import if any
        if 'agent_main' in sys.modules:
            del sys.modules['agent_main']
        
        # Read source
        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        with open(agent_main_path, 'r') as f:
            source_code = f.read()
        
        # Prepare the module with required globals
        mod = types.ModuleType('agent_main')
        mod.__file__ = agent_main_path
        
        # Inject logging
        mod.__dict__['logging'] = logging
        
        # Try to find and load CONFIG
        # Look for a config file or define a default
        config_candidates = [
            os.path.join(os.path.dirname(agent_main_path), 'config.json'),
            os.path.join(os.path.dirname(agent_main_path), 'config.yaml'),
            os.path.join(os.path.dirname(agent_main_path), 'config.py'),
        ]
        
        CONFIG = None
        for cfg_path in config_candidates:
            if os.path.exists(cfg_path):
                if cfg_path.endswith('.json'):
                    import json
                    with open(cfg_path, 'r') as f:
                        CONFIG = json.load(f)
                    break
                elif cfg_path.endswith('.py'):
                    # Try to import config module
                    try:
                        spec = importlib.util.spec_from_file_location("config_module", cfg_path)
                        config_mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(config_mod)
                        if hasattr(config_mod, 'CONFIG'):
                            CONFIG = config_mod.CONFIG
                        break
                    except Exception:
                        pass
        
        # If CONFIG not found from files, try to find it in the source or set defaults
        if CONFIG is None:
            # Look for CONFIG in existing modules or define defaults
            # Try importing from common locations
            try:
                sys.path.insert(0, os.path.dirname(agent_main_path))
                from config import CONFIG as _cfg
                CONFIG = _cfg
            except Exception:
                pass
        
        if CONFIG is None:
            # Check if there's a config.yml or similar
            try:
                import yaml
                yaml_path = os.path.join(os.path.dirname(agent_main_path), 'config.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        CONFIG = yaml.safe_load(f)
            except Exception:
                pass
        
        if CONFIG is None:
            # Try config.yml
            try:
                import yaml
                yml_path = os.path.join(os.path.dirname(agent_main_path), 'config.yml')
                if os.path.exists(yml_path):
                    with open(yml_path, 'r') as f:
                        CONFIG = yaml.safe_load(f)
            except Exception:
                pass
        
        if CONFIG is None:
            # Search for any .json file that might be config
            dir_path = os.path.dirname(agent_main_path)
            for fname in os.listdir(dir_path):
                if fname.endswith('.json') and 'config' in fname.lower():
                    try:
                        import json
                        with open(os.path.join(dir_path, fname), 'r') as f:
                            CONFIG = json.load(f)
                        break
                    except Exception:
                        pass
        
        if CONFIG is None:
            # Last resort: try to find CONFIG defined anywhere in the directory
            dir_path = os.path.dirname(agent_main_path)
            for fname in os.listdir(dir_path):
                if fname.endswith('.py') and fname != 'agent_main.py' and fname != 'test_main.py':
                    try:
                        fpath = os.path.join(dir_path, fname)
                        with open(fpath, 'r') as f:
                            content = f.read()
                        if 'CONFIG' in content:
                            spec = importlib.util.spec_from_file_location(fname[:-3], fpath)
                            temp_mod = importlib.util.module_from_spec(spec)
                            temp_mod.__dict__['logging'] = logging
                            try:
                                spec.loader.exec_module(temp_mod)
                                if hasattr(temp_mod, 'CONFIG'):
                                    CONFIG = temp_mod.CONFIG
                                    break
                            except Exception:
                                pass
                    except Exception:
                        pass

        if CONFIG is None:
            # Try loading from a .toml file
            dir_path = os.path.dirname(agent_main_path)
            for fname in os.listdir(dir_path):
                if fname.endswith('.toml'):
                    try:
                        import toml
                        with open(os.path.join(dir_path, fname), 'r') as f:
                            CONFIG = toml.load(f)
                        break
                    except Exception:
                        pass

        if CONFIG is not None:
            mod.__dict__['CONFIG'] = CONFIG
        
        # Register the module before executing
        sys.modules['agent_main'] = mod
        
        # Execute the source in the module's namespace
        exec(compile(source_code, agent_main_path, 'exec'), mod.__dict__)
        
except Exception as e:
    print(f"Error during module setup: {e}")
    traceback.print_exc()

# Now try the standard import
try:
    from agent_main import main
except Exception as e:
    print(f"FATAL: Could not import 'main' from 'agent_main': {e}")
    traceback.print_exc()
    sys.exit(1)

# Import verification utility
try:
    from verification_utils import recursive_check
except Exception as e:
    print(f"FATAL: Could not import 'recursive_check' from 'verification_utils': {e}")
    traceback.print_exc()
    sys.exit(1)


def test_main():
    """Test the main function using captured data."""

    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_DS_DC1_sandbox/run_code/std_data/data_main.pkl'
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
        print("FATAL: No outer data file (data_main.pkl) found in data_paths.")
        sys.exit(1)

    # -------------------------------------------------------
    # Phase 1: Load outer data and run main()
    # -------------------------------------------------------
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FATAL: Could not load outer data file: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'N/A')}")
    print(f"Outer args count: {len(outer_args)}, Outer kwargs keys: {list(outer_kwargs.keys())}")

    # -------------------------------------------------------
    # Determine scenario
    # -------------------------------------------------------
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        print(f"Found {len(inner_paths)} inner data file(s).")

        # Run main to get the operator/closure
        try:
            print("Running main(*outer_args, **outer_kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FATAL: main() raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"WARNING: agent_operator is not callable (type={type(agent_operator)}). "
                  f"Attempting to use it as result directly.")

        # Process each inner path
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Inner test {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FATAL: Could not load inner data file: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            print(f"Inner args count: {len(inner_args)}, Inner kwargs keys: {list(inner_kwargs.keys())}")

            try:
                print("Running agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FATAL: agent_operator raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare
            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FATAL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED inner test {idx + 1}: {msg}")
                all_passed = False
            else:
                print(f"PASSED inner test {idx + 1}")

        if not all_passed:
            print("\nTEST FAILED: One or more inner tests did not pass.")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            print("Running main(*outer_args, **outer_kwargs)...")
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FATAL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected)}")

        # Compare
        try:
            passed, msg = recursive_check(expected, result)
        except Exception as e:
            print(f"FATAL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == "__main__":
    test_main()