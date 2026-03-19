import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to inject a dummy DhariwalUNet before importing agent_main
# since agent_main references it at module level in _model_dict
import types

# Create a dummy DhariwalUNet class that can be used as a placeholder
# The actual model will be loaded from checkpoint via pickle/dill
class DummyDhariwalUNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
    def forward(self, x, *args, **kwargs):
        return x

# Inject DhariwalUNet into builtins so it's available when agent_main is parsed
import builtins
builtins.DhariwalUNet = DummyDhariwalUNet

# Also try to make it available in the agent_main module's namespace
# by pre-creating the module and injecting the name
import importlib
agent_main_spec = importlib.util.spec_from_file_location(
    "agent_main", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_main.py")
)
# We need to patch the global namespace before the module loads
# Let's do it by modifying sys.modules approach

# Alternative: directly modify the source or use exec approach
# Simplest: just add DhariwalUNet to the globals before import

# Create a temporary module-level injection
_original_import = builtins.__import__

def _patched_import(name, *args, **kwargs):
    mod = _original_import(name, *args, **kwargs)
    if name == 'agent_main' or (hasattr(mod, '__name__') and getattr(mod, '__name__', '') == 'agent_main'):
        if not hasattr(mod, 'DhariwalUNet'):
            mod.DhariwalUNet = DummyDhariwalUNet
    return mod

# Instead of patching import, let's just write DhariwalUNet into the agent_main module namespace
# by loading it manually

def load_agent_main():
    """Load agent_main with DhariwalUNet injected."""
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_main.py")
    
    with open(module_path, 'r') as f:
        source = f.read()
    
    # Create module
    mod = types.ModuleType('agent_main')
    mod.__file__ = module_path
    mod.__name__ = 'agent_main'
    
    # Set up the module's namespace with DhariwalUNet
    mod.__dict__['DhariwalUNet'] = DummyDhariwalUNet
    
    # Also need __builtins__
    mod.__dict__['__builtins__'] = builtins.__dict__
    
    # Execute the module code in its namespace
    code = compile(source, module_path, 'exec')
    exec(code, mod.__dict__)
    
    # Register in sys.modules
    sys.modules['agent_main'] = mod
    
    return mod


from verification_utils import recursive_check


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_score_mri_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (main) and inner (parent_function) paths
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

    # Phase 1: Load outer data
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
    expected_output = outer_data.get('output', None)

    print(f"Outer data func_name: {outer_data.get('func_name', 'unknown')}")
    print(f"Outer args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Load agent_main with DhariwalUNet injected
    print("Loading agent_main module with DhariwalUNet injected...")
    try:
        agent_module = load_agent_main()
        target_main = agent_module.main
    except Exception as e:
        print(f"ERROR: Failed to load agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Scenario B detected: Factory/Closure pattern")

        # Run main to get operator
        print("Running main(*args, **kwargs) to get operator...")
        try:
            agent_operator = target_main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: Expected callable operator from main(), got {type(agent_operator)}")
            sys.exit(1)

        # Process each inner path
        all_passed = True
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"Inner data func_name: {inner_data.get('func_name', 'unknown')}")
            print(f"Inner args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            # Execute operator with inner data
            print("Executing agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAILED for inner path {os.path.basename(inner_path)}: {msg}")
                all_passed = False
            else:
                print(f"PASSED for inner path {os.path.basename(inner_path)}")

        if all_passed:
            print("\nTEST PASSED")
            sys.exit(0)
        else:
            print("\nTEST FAILED")
            sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Scenario A detected: Simple function call")

        # The main() function in this case returns None (it's a script-like function)
        # We need to run it and compare the output
        print("Running main(*args, **kwargs)...")
        try:
            actual_result = target_main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare results
        print("Comparing results...")
        print(f"Expected output type: {type(expected_output)}")
        print(f"Actual result type: {type(actual_result)}")

        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)


if __name__ == '__main__':
    main()