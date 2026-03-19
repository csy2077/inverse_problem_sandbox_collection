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

        # Fix data_dir to the actual found path (make it relative to SCRIPT_DIR in agent_main)
        # The agent_main uses rel_path(data_dir, filename) = os.path.join(SCRIPT_DIR, data_dir, filename)
        # So we need data_dir such that os.path.join(SCRIPT_DIR_of_agent, data_dir, 'DC1.mat') works
        
        # Read agent_main.py to find its SCRIPT_DIR
        agent_main_path = os.path.join(script_dir, 'agent_main.py')
        if os.path.isfile(agent_main_path):
            # SCRIPT_DIR in agent_main will be the directory of agent_main.py
            agent_script_dir = os.path.dirname(os.path.abspath(agent_main_path))
            
            if actual_data_dir is not None:
                # Compute relative path from agent_script_dir to actual_data_dir
                rel_data_dir = os.path.relpath(actual_data_dir, agent_script_dir)
                CONFIG["data_dir"] = rel_data_dir
                print(f"Set CONFIG['data_dir'] = '{rel_data_dir}'")
            
            # Also fix figs_dir to a writable temp location
            figs_dir = os.path.join(script_dir, 'figs_test_output')
            os.makedirs(figs_dir, exist_ok=True)
            CONFIG["figs_dir"] = os.path.relpath(figs_dir, agent_script_dir)
            print(f"Set CONFIG['figs_dir'] = '{CONFIG['figs_dir']}'")

        builtins.CONFIG = CONFIG

        from agent_main import main
        print("Successfully imported main from agent_main")
    except Exception as e:
        print(f"FAIL: Could not import main from agent_main: {e}")
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