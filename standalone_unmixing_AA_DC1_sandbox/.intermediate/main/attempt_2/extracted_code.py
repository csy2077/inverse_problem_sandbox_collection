import sys
import os
import dill
import numpy as np
import logging
import traceback
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_agent_main():
    """Patch agent_main.py to inject CONFIG and logging if needed."""
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    
    if not os.path.exists(agent_main_path):
        return None, False
    
    with open(agent_main_path, 'r') as f:
        original_content = f.read()
    
    new_content = original_content
    needs_patch = False
    
    # Check if 'import logging' is missing
    if 'import logging' not in new_content:
        new_content = 'import logging\n' + new_content
        needs_patch = True
    
    # Check if CONFIG is used but not defined
    if 'CONFIG' in new_content:
        # Check if CONFIG is properly defined
        has_config_def = False
        lines = new_content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if re.match(r'^CONFIG\s*=', stripped):
                has_config_def = True
                break
            if 'import' in stripped and 'CONFIG' in stripped:
                has_config_def = True
                break
        
        if not has_config_def:
            # Try to find config file
            search_dir = os.path.dirname(agent_main_path)
            config_data = None
            
            # Search for config files
            for root, dirs, files in os.walk(search_dir):
                for fname in files:
                    if fname.endswith('.json'):
                        try:
                            fpath = os.path.join(root, fname)
                            with open(fpath, 'r') as f:
                                candidate = json.load(f)
                            if isinstance(candidate, dict) and ('seed' in candidate or 'EPS' in candidate or 'dataset' in candidate):
                                config_data = candidate
                                print(f"Found config at: {fpath}")
                                break
                        except:
                            pass
                if config_data is not None:
                    break
            
            # Also check parent directories
            if config_data is None:
                for parent_dir in [os.path.join(search_dir, '..'), os.path.join(search_dir, 'run_code')]:
                    parent_dir = os.path.abspath(parent_dir)
                    if os.path.isdir(parent_dir):
                        for fname in os.listdir(parent_dir):
                            if fname.endswith('.json'):
                                try:
                                    fpath = os.path.join(parent_dir, fname)
                                    with open(fpath, 'r') as f:
                                        candidate = json.load(f)
                                    if isinstance(candidate, dict) and ('seed' in candidate or 'EPS' in candidate or 'dataset' in candidate):
                                        config_data = candidate
                                        print(f"Found config at: {fpath}")
                                        break
                                except:
                                    pass
                    if config_data is not None:
                        break
            
            if config_data is None:
                config_data = {
                    "seed": 0,
                    "SNR": 30,
                    "dataset": "DC1",
                    "data_dir": "./data",
                    "figs_dir": "./figs",
                    "l2_normalization": False,
                    "projection": True,
                    "EPS": 1e-6,
                    "model": {
                        "epsilon": 1e-3,
                        "robust": False,
                        "computeXtX": True,
                        "stepsFISTA": 3,
                        "stepsAS": 50,
                        "randominit": True,
                        "numThreads": -1
                    }
                }
                print(f"WARNING: Using default CONFIG")
            
            # We need to inject CONFIG BEFORE any usage. The issue is that
            # EPS = CONFIG["EPS"] happens at module level, so CONFIG must be
            # defined before that line.
            
            # Strategy: Find the exact line where CONFIG is first used and insert before it
            config_str = json.dumps(config_data)
            # Escape any single quotes in the JSON string
            config_str = config_str.replace("'", "\\'")
            
            config_inject = f"import json as _json_cfg_inject_\nCONFIG = _json_cfg_inject_.loads('{config_str}')\n"
            
            # Find the first line that references CONFIG (non-import, non-comment)
            lines = new_content.split('\n')
            insert_idx = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                if 'CONFIG' in stripped and not stripped.startswith('import') and not stripped.startswith('from'):
                    insert_idx = i
                    break
            
            if insert_idx is not None:
                lines.insert(insert_idx, config_inject)
                new_content = '\n'.join(lines)
                needs_patch = True
            else:
                # Insert after all imports
                insert_after = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_after = i + 1
                lines.insert(insert_after, config_inject)
                new_content = '\n'.join(lines)
                needs_patch = True
    
    if needs_patch:
        with open(agent_main_path, 'w') as f:
            f.write(new_content)
        print("Patched agent_main.py")
        return original_content, True
    
    return original_content, False


def restore_agent_main(original_content):
    """Restore agent_main.py to original content."""
    if original_content is not None:
        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        with open(agent_main_path, 'w') as f:
            f.write(original_content)
        print("Restored agent_main.py")


def test_main():
    """Test the main function using captured data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_AA_DC1_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Separate outer and inner paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    assert outer_path is not None, f"Could not find outer data file in {data_paths}"
    
    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}")
    print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
    
    # Patch agent_main.py before importing
    original_content, patched = patch_agent_main()
    
    # Import main
    try:
        if 'agent_main' in sys.modules:
            del sys.modules['agent_main']
        from agent_main import main
    except Exception as e:
        print(f"FAILED to import main from agent_main: {e}")
        traceback.print_exc()
        if patched:
            restore_agent_main(original_content)
        sys.exit(1)
    
    from verification_utils import recursive_check
    
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")
        
        try:
            print("Running main(*args, **kwargs) to get operator...")
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            if patched:
                restore_agent_main(original_content)
            sys.exit(1)
        
        assert callable(agent_operator), f"Expected callable operator, got {type(agent_operator)}"
        print(f"Got callable operator: {type(agent_operator)}")
        
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAILED to load inner data: {e}")
                traceback.print_exc()
                if patched:
                    restore_agent_main(original_content)
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                print("Running agent_operator(*inner_args, **inner_kwargs)...")
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED to execute operator: {e}")
                traceback.print_exc()
                if patched:
                    restore_agent_main(original_content)
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                if patched:
                    restore_agent_main(original_content)
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                if patched:
                    restore_agent_main(original_content)
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        if patched:
            restore_agent_main(original_content)
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")
        
        try:
            print("Running main(*args, **kwargs)...")
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main: {e}")
            traceback.print_exc()
            if patched:
                restore_agent_main(original_content)
            sys.exit(1)
        
        if patched:
            restore_agent_main(original_content)
        
        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAILED during comparison: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == "__main__":
    test_main()