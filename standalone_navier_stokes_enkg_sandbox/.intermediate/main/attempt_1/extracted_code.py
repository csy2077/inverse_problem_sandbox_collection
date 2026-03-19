import sys
import os
import dill
import torch
import numpy as np
import traceback

# We need to find and import DhariwalUNet before importing agent_main
# Search for it in the project directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try to find DhariwalUNet from various possible locations
def find_and_import_dhariwal():
    """Try to find and import DhariwalUNet from the project structure."""
    import importlib
    import glob
    
    # Common module names that might contain DhariwalUNet
    possible_modules = [
        'torch_utils',
        'training',
        'networks',
        'model',
        'models',
        'unet',
        'dhariwal',
        'guided_diffusion',
        'training.networks',
    ]
    
    # Add script_dir and parent dirs to path
    for d in [script_dir, os.path.join(script_dir, '..'), os.path.join(script_dir, 'data_standalone')]:
        if d not in sys.path:
            sys.path.insert(0, d)
    
    # Try importing from common module names
    for mod_name in possible_modules:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, 'DhariwalUNet'):
                return getattr(mod, 'DhariwalUNet')
        except:
            pass
    
    # Search .py files for DhariwalUNet class definition
    search_dirs = [script_dir, os.path.join(script_dir, '..')]
    for search_dir in search_dirs:
        for py_file in glob.glob(os.path.join(search_dir, '**', '*.py'), recursive=True):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                if 'class DhariwalUNet' in content:
                    # Get module name from file path
                    rel_path = os.path.relpath(py_file, script_dir)
                    mod_name = rel_path.replace(os.sep, '.').replace('.py', '')
                    try:
                        mod = importlib.import_module(mod_name)
                        if hasattr(mod, 'DhariwalUNet'):
                            return getattr(mod, 'DhariwalUNet')
                    except:
                        pass
                    # Try with parent dir
                    rel_path2 = os.path.relpath(py_file, os.path.join(script_dir, '..'))
                    mod_name2 = rel_path2.replace(os.sep, '.').replace('.py', '')
                    try:
                        mod = importlib.import_module(mod_name2)
                        if hasattr(mod, 'DhariwalUNet'):
                            return getattr(mod, 'DhariwalUNet')
                    except:
                        pass
            except:
                pass
    
    # Try to find in pickle/dill files
    for search_dir in search_dirs:
        for pkl_file in glob.glob(os.path.join(search_dir, '**', '*.pkl'), recursive=True):
            try:
                with open(pkl_file, 'rb') as f:
                    data = dill.load(f)
                if isinstance(data, dict) and 'ema' in data:
                    # This might be a checkpoint with the model
                    return type(data['ema'])
            except:
                pass
    
    return None

# Try to find DhariwalUNet and inject into builtins/globals before import
DhariwalUNet_cls = find_and_import_dhariwal()

if DhariwalUNet_cls is None:
    # Try loading from the checkpoint to get the class
    # First, try to load it via pickle which may have the class embedded
    try:
        config_path = os.path.join(script_dir, 'data_standalone', 'standalone_navier_stokes_enkg.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            prior_path = config.get('problem', {}).get('prior', '')
            if not os.path.isabs(prior_path):
                prior_path = os.path.join(script_dir, prior_path)
            if os.path.exists(prior_path):
                import pickle
                with open(prior_path, 'rb') as f:
                    ckpt = pickle.load(f)
                if 'ema' in ckpt:
                    net = ckpt['ema']
                    # Extract the model class
                    if hasattr(net, 'model'):
                        DhariwalUNet_cls = type(net.model)
                    del ckpt
    except:
        pass

if DhariwalUNet_cls is None:
    # As a last resort, create a stub or try torch_utils.persistence
    try:
        # Many EDM codebases use torch_utils
        sys.path.insert(0, script_dir)
        # Check if there's a networks module
        import importlib
        for candidate in ['networks', 'training.networks', 'torch_utils.networks']:
            try:
                mod = importlib.import_module(candidate)
                if hasattr(mod, 'DhariwalUNet'):
                    DhariwalUNet_cls = getattr(mod, 'DhariwalUNet')
                    break
            except:
                continue
    except:
        pass

if DhariwalUNet_cls is None:
    # Try to unpickle the checkpoint with dill which might resolve the class
    try:
        config_path = os.path.join(script_dir, 'data_standalone', 'standalone_navier_stokes_enkg.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            prior_path = config.get('problem', {}).get('prior', '')
            if not os.path.isabs(prior_path):
                prior_path = os.path.join(script_dir, prior_path)
            if os.path.exists(prior_path):
                with open(prior_path, 'rb') as f:
                    ckpt = dill.load(f)
                if 'ema' in ckpt:
                    net = ckpt['ema']
                    if hasattr(net, 'model'):
                        DhariwalUNet_cls = type(net.model)
                del ckpt
    except:
        pass

# Inject DhariwalUNet into the agent_main module's namespace
# We need to make it available before agent_main is imported
import builtins
if DhariwalUNet_cls is not None:
    builtins.DhariwalUNet = DhariwalUNet_cls
    print(f"Found DhariwalUNet: {DhariwalUNet_cls}")
else:
    # Create a dummy/placeholder - the actual model loading might use pickle which has the real class
    # We need to handle the case where the _model_dict reference fails at module level
    # Patch the agent_main source to handle missing DhariwalUNet gracefully
    
    # Read agent_main.py and check if DhariwalUNet is used only in dict
    agent_main_path = os.path.join(script_dir, 'agent_main.py')
    if os.path.exists(agent_main_path):
        with open(agent_main_path, 'r') as f:
            source = f.read()
        
        # Check if we can find the actual import or definition
        # Look for any import statement that brings in DhariwalUNet
        import re
        imports = re.findall(r'from\s+(\S+)\s+import.*DhariwalUNet', source)
        if imports:
            for imp_mod in imports:
                try:
                    mod = __import__(imp_mod, fromlist=['DhariwalUNet'])
                    DhariwalUNet_cls = getattr(mod, 'DhariwalUNet')
                    builtins.DhariwalUNet = DhariwalUNet_cls
                    print(f"Found DhariwalUNet from import: {imp_mod}")
                    break
                except:
                    continue
    
    if DhariwalUNet_cls is None:
        # Create a placeholder class that will work for module-level dict creation
        # The actual model will be loaded from checkpoint via pickle
        print("WARNING: Could not find DhariwalUNet, creating placeholder")
        
        class DhariwalUNetPlaceholder(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.kwargs = kwargs
            def forward(self, *args, **kwargs):
                raise NotImplementedError("DhariwalUNet placeholder - should not be called directly")
        
        builtins.DhariwalUNet = DhariwalUNetPlaceholder
        DhariwalUNet_cls = DhariwalUNetPlaceholder

# Now try the import
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_enkg_sandbox/run_code/std_data/data_main.pkl']

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
    print("ERROR: Could not find outer data file (data_main.pkl)")
    sys.exit(1)

# Load outer data
try:
    with open(outer_path, 'rb') as f:
        outer_data = dill.load(f)
    print(f"Successfully loaded outer data from: {outer_path}")
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'N/A')}")
except Exception as e:
    print(f"ERROR loading outer data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Extract outer args and kwargs
outer_args = outer_data.get('args', ())
outer_kwargs = outer_data.get('kwargs', {})
expected_output = outer_data.get('output', None)

print(f"Outer args count: {len(outer_args)}")
print(f"Outer kwargs keys: {list(outer_kwargs.keys()) if isinstance(outer_kwargs, dict) else 'N/A'}")
print(f"Expected output type: {type(expected_output)}")

# Scenario A: Simple function call (no inner paths)
if len(inner_paths) == 0:
    print("\n=== Scenario A: Simple Function Call ===")
    try:
        from agent_main import main
        print("Successfully imported main from agent_main")
    except Exception as e:
        print(f"ERROR importing main: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("Running main(*args, **kwargs)...")
        result = main(*outer_args, **outer_kwargs)
        print(f"main() returned: {type(result)}")
    except Exception as e:
        print(f"ERROR running main(): {e}")
        traceback.print_exc()
        sys.exit(1)

    # Compare result with expected output
    try:
        from verification_utils import recursive_check
        passed, msg = recursive_check(expected_output, result)
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)

else:
    # Scenario B: Factory/Closure Pattern
    print("\n=== Scenario B: Factory/Closure Pattern ===")
    try:
        from agent_main import main
        print("Successfully imported main from agent_main")
    except Exception as e:
        print(f"ERROR importing main: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 1: Reconstruct operator
    try:
        print("Running main(*outer_args, **outer_kwargs) to get operator...")
        agent_operator = main(*outer_args, **outer_kwargs)
        print(f"Got operator of type: {type(agent_operator)}")
        if not callable(agent_operator):
            print(f"WARNING: Returned operator is not callable (type: {type(agent_operator)})")
    except Exception as e:
        print(f"ERROR creating operator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 2: Execute with inner data
    all_passed = True
    for inner_path in inner_paths:
        print(f"\nProcessing inner data: {inner_path}")
        try:
            with open(inner_path, 'rb') as f:
                inner_data = dill.load(f)
            print(f"Inner data keys: {list(inner_data.keys())}")
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)

        inner_args = inner_data.get('args', ())
        inner_kwargs = inner_data.get('kwargs', {})
        inner_expected = inner_data.get('output', None)

        try:
            print("Executing operator with inner args...")
            result = agent_operator(*inner_args, **inner_kwargs)
            print(f"Operator returned: {type(result)}")
        except Exception as e:
            print(f"ERROR executing operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            from verification_utils import recursive_check
            passed, msg = recursive_check(inner_expected, result)
            if passed:
                print(f"Inner test PASSED for {inner_path}")
            else:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)