import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to handle the missing DhariwalUNet before importing agent_main.
# First, try to find and import it from the project structure.
def _find_and_patch_dhariwal():
    """Try to find DhariwalUNet and make it available globally before agent_main imports."""
    # Common locations where DhariwalUNet might be defined
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InverseBench'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    ]
    
    # Try importing from known module paths
    import_attempts = [
        'networks',
        'model',
        'models',
        'InverseBench.networks',
        'InverseBench.model',
        'InverseBench.models',
        'models.networks',
        'guided_diffusion.unet',
        'src.networks',
        'src.models',
    ]
    
    for mod_name in import_attempts:
        try:
            mod = __import__(mod_name, fromlist=['DhariwalUNet'])
            if hasattr(mod, 'DhariwalUNet'):
                return mod.DhariwalUNet
        except ImportError:
            continue
    
    # Try searching python files for class definition
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for fname in files:
                if fname.endswith('.py') and fname != 'test_main.py' and fname != 'agent_main.py':
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r') as f:
                            content = f.read()
                        if 'class DhariwalUNet' in content:
                            # Try to import this module
                            rel_path = os.path.relpath(fpath, os.path.dirname(os.path.abspath(__file__)))
                            mod_path = rel_path.replace(os.sep, '.').replace('.py', '')
                            try:
                                mod = __import__(mod_path, fromlist=['DhariwalUNet'])
                                if hasattr(mod, 'DhariwalUNet'):
                                    return mod.DhariwalUNet
                            except Exception:
                                # Try exec approach
                                import importlib.util
                                spec = importlib.util.spec_from_file_location("_dhariwal_module", fpath)
                                if spec and spec.loader:
                                    mod = importlib.util.module_from_spec(spec)
                                    try:
                                        spec.loader.exec_module(mod)
                                        if hasattr(mod, 'DhariwalUNet'):
                                            return mod.DhariwalUNet
                                    except Exception:
                                        continue
                    except Exception:
                        continue
    
    return None

# Try to find DhariwalUNet
_DhariwalUNet = _find_and_patch_dhariwal()

if _DhariwalUNet is not None:
    import builtins
    builtins.DhariwalUNet = _DhariwalUNet
else:
    # If we can't find it, we need to create a stub or try to make the import work
    # by injecting into the module namespace that agent_main will use
    # Let's try a different approach: modify agent_main's globals after partial import
    pass

# Now try to handle the import with patching
try:
    # First attempt: direct import (might work if DhariwalUNet was found above)
    from agent_main import main
except NameError as e:
    if 'DhariwalUNet' in str(e):
        # Need to patch the module. Read agent_main.py source and exec with DhariwalUNet defined
        import importlib
        import types
        
        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        
        # Try harder to find DhariwalUNet from dill-pickled data
        # It might be serialized in the data file
        try:
            data_path = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_main.pkl'
            with open(data_path, 'rb') as f:
                _temp_data = dill.load(f)
            # Check if DhariwalUNet is accessible from loaded objects
            del _temp_data
        except Exception:
            pass
        
        # Create a minimal DhariwalUNet stub that can be used for the _model_dict
        # Since the actual model weights are loaded from checkpoint, we need the real class
        # Let's search more aggressively
        
        # Search in all sys.path locations
        for p in sys.path:
            if not os.path.isdir(p):
                continue
            for root, dirs, files in os.walk(p):
                for fname in files:
                    if fname.endswith('.py'):
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, 'r') as f:
                                first_bytes = f.read(50000)
                            if 'class DhariwalUNet' in first_bytes:
                                import importlib.util
                                spec = importlib.util.spec_from_file_location("_dhu_mod", fpath)
                                if spec and spec.loader:
                                    mod = importlib.util.module_from_spec(spec)
                                    try:
                                        spec.loader.exec_module(mod)
                                        if hasattr(mod, 'DhariwalUNet'):
                                            _DhariwalUNet = mod.DhariwalUNet
                                            import builtins
                                            builtins.DhariwalUNet = _DhariwalUNet
                                            break
                                    except Exception:
                                        continue
                        except Exception:
                            continue
                if _DhariwalUNet is not None:
                    break
            if _DhariwalUNet is not None:
                break
        
        if _DhariwalUNet is None:
            # Last resort: try to load it via the pickle/dill mechanism
            # or create a placeholder that will work with torch.load
            print("WARNING: Could not find DhariwalUNet class. Attempting to create module with placeholder.")
            
            # Read agent_main.py and find what DhariwalUNet needs
            # Since the code does torch.load for weights, the class structure matters
            # But for testing main() which returns None, we might be able to use a mock
            
            # Try to find it in site-packages or pip installed packages
            try:
                from guided_diffusion.unet import UNetModel as DhariwalUNet
                import builtins
                builtins.DhariwalUNet = DhariwalUNet
            except ImportError:
                pass
        
        # Retry import
        try:
            from agent_main import main
        except NameError:
            # Final attempt: manually load and patch
            agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
            with open(agent_main_path, 'r') as f:
                source = f.read()
            
            # We need to find DhariwalUNet from somewhere in the codebase
            # Search recursively from project root
            project_root = os.path.dirname(os.path.abspath(__file__))
            found_class = None
            
            for root, dirs, files in os.walk(project_root):
                # Skip hidden dirs, __pycache__, etc
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for fname in files:
                    if not fname.endswith('.py'):
                        continue
                    if fname in ('test_main.py', 'agent_main.py'):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r') as f:
                            content = f.read()
                        if 'class DhariwalUNet' in content:
                            # Found it! Add its directory to path and import
                            dir_of_file = os.path.dirname(fpath)
                            if dir_of_file not in sys.path:
                                sys.path.insert(0, dir_of_file)
                            mod_name = fname[:-3]
                            mod = __import__(mod_name, fromlist=['DhariwalUNet'])
                            if hasattr(mod, 'DhariwalUNet'):
                                found_class = mod.DhariwalUNet
                                break
                    except Exception:
                        continue
                if found_class is not None:
                    break
            
            if found_class is not None:
                import builtins
                builtins.DhariwalUNet = found_class
                # Clear cached module and reimport
                if 'agent_main' in sys.modules:
                    del sys.modules['agent_main']
                from agent_main import main
            else:
                print("FATAL: Cannot find DhariwalUNet class anywhere in the project.")
                print("Attempting to load agent_main with exec and inject DhariwalUNet as a dummy...")
                
                # As absolute last resort, try to exec agent_main with a DhariwalUNet 
                # that we extract from the pickle checkpoint
                # The main() function loads from 'weights/inv-scatter-5m.pt' 
                # which uses torch.load/pickle.load, so it needs the real class
                
                # Check if there's a networks.py or similar in common locations
                common_paths = [
                    '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox',
                    '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/InverseBench',
                    '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/src',
                ]
                for cp in common_paths:
                    if cp not in sys.path and os.path.isdir(cp):
                        sys.path.insert(0, cp)
                
                # One more try with expanded path
                try:
                    if 'agent_main' in sys.modules:
                        del sys.modules['agent_main']
                    from agent_main import main
                except Exception as final_e:
                    print(f"FATAL: Cannot import agent_main: {final_e}")
                    traceback.print_exc()
                    sys.exit(1)
    else:
        raise

from verification_utils import recursive_check


def main_test():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_reddiff_sandbox/run_code/std_data/data_main.pkl'
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

    # Load outer data
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

    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    if len(inner_paths) > 0:
        # ============================================================
        # Scenario B: Factory/Closure Pattern
        # ============================================================
        print("Detected Scenario B (Factory/Closure pattern).")

        # Phase 1: Reconstruct operator
        print("Phase 1: Running main(*outer_args, **outer_kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main() to get operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"ERROR: main() did not return a callable. Got type: {type(agent_operator)}")
            sys.exit(1)

        print(f"Phase 1 complete. Got operator of type: {type(agent_operator)}")

        # Phase 2: Execute with inner data
        for inner_path in sorted(inner_paths):
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

            print(f"Inner data loaded. func_name={inner_data.get('func_name', 'N/A')}")
            print(f"  args count: {len(inner_args)}, kwargs keys: {list(inner_kwargs.keys())}")

            print("Phase 2: Running agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")

        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # ============================================================
        # Scenario A: Simple Function
        # ============================================================
        print("Detected Scenario A (Simple function call).")

        print("Running main(*outer_args, **outer_kwargs)...")
        try:
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Comparison
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main_test()