import sys
import os
import dill
import numpy as np
import traceback

# Ensure scipy is imported and available in builtins before agent_main is imported
import scipy
import scipy.sparse
import scipy.sparse.linalg
import builtins
builtins.scipy = scipy

# Patch the agent_main module's globals before import
# We need to inject scipy into the module's namespace
import types

# First, ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read agent_main.py source and inject scipy import at the top
agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
with open(agent_main_path, 'r') as f:
    source = f.read()

# Check if scipy import is missing and add it
if 'import scipy' not in source:
    source = "import scipy\nimport scipy.sparse\nimport scipy.sparse.linalg\n" + source

# Create the module manually
agent_main_module = types.ModuleType('agent_main')
agent_main_module.__file__ = agent_main_path
agent_main_module.__name__ = 'agent_main'
sys.modules['agent_main'] = agent_main_module

try:
    exec(compile(source, agent_main_path, 'exec'), agent_main_module.__dict__)
except Exception as e:
    print(f"ERROR: Failed to load agent_main: {e}")
    traceback.print_exc()
    sys.exit(1)

main = agent_main_module.main

from verification_utils import recursive_check


def test_main():
    """Test the main function against captured standard data."""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data/data_main.pkl'
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
    
    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)
    
    # --- Phase 1: Load outer data and run main ---
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)
    
    print(f"Outer data function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")
    
    try:
        print("Running main(*args, **kwargs)...")
        agent_result = main(*outer_args, **outer_kwargs)
        print(f"main() returned: {type(agent_result)}")
    except Exception as e:
        print(f"ERROR: main() execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # --- Phase 2: Determine scenario and verify ---
    if inner_paths:
        # Scenario B: Factory/Closure pattern
        print(f"\n--- Scenario B: Factory/Closure pattern detected ---")
        print(f"Found {len(inner_paths)} inner data file(s).")
        
        # Verify agent_result is callable
        if not callable(agent_result):
            print(f"ERROR: Expected main() to return a callable, got {type(agent_result)}")
            sys.exit(1)
        
        all_passed = True
        for idx, inner_path in enumerate(inner_paths):
            print(f"\n--- Testing inner data file {idx + 1}/{len(inner_paths)}: {os.path.basename(inner_path)} ---")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")
            
            try:
                actual_result = agent_result(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Operator execution failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"FAILED: Inner test {idx + 1}: {msg}")
                all_passed = False
            else:
                print(f"PASSED: Inner test {idx + 1}")
        
        if not all_passed:
            print("\nTEST FAILED")
            sys.exit(1)
        else:
            print("\nTEST PASSED")
            sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print(f"\n--- Scenario A: Simple function pattern ---")
        
        expected = outer_output
        actual_result = agent_result
        
        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"FAILED: {msg}")
            print("\nTEST FAILED")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    test_main()