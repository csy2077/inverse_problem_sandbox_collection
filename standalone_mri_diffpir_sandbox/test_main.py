import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to handle the DhariwalUNet import issue before importing agent_main
# First, try to find and make DhariwalUNet available
try:
    # Try importing from common locations
    from torch_utils import persistence
except ImportError:
    pass

# Try to find DhariwalUNet in various possible modules
_dhariwal_unet_class = None

def _find_dhariwal_unet():
    global _dhariwal_unet_class
    # Try various import paths
    possible_modules = [
        ('networks', 'DhariwalUNet'),
        ('torch_utils.networks', 'DhariwalUNet'),
        ('training.networks', 'DhariwalUNet'),
        ('model', 'DhariwalUNet'),
        ('models', 'DhariwalUNet'),
        ('models.networks', 'DhariwalUNet'),
        ('unet', 'DhariwalUNet'),
        ('diffusion', 'DhariwalUNet'),
    ]
    for mod_name, cls_name in possible_modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                _dhariwal_unet_class = cls
                return cls
        except ImportError:
            continue
    
    # Search recursively in the current directory
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for fname in files:
            if fname.endswith('.py') and fname != 'test_main.py':
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    if 'class DhariwalUNet' in content:
                        # Derive module name
                        relpath = os.path.relpath(fpath, os.path.dirname(os.path.abspath(__file__)))
                        mod_name = relpath.replace(os.sep, '.').replace('.py', '')
                        mod = __import__(mod_name, fromlist=['DhariwalUNet'])
                        cls = getattr(mod, 'DhariwalUNet', None)
                        if cls is not None:
                            _dhariwal_unet_class = cls
                            return cls
                except Exception:
                    continue
    return None

_find_dhariwal_unet()

# Inject DhariwalUNet into builtins so agent_main can find it
if _dhariwal_unet_class is not None:
    import builtins
    builtins.DhariwalUNet = _dhariwal_unet_class

# If we still don't have it, we need to patch agent_main's module-level code
# by injecting it into the agent_main module's namespace before it fully loads
if _dhariwal_unet_class is None:
    # Try to read agent_main.py and find what module DhariwalUNet might come from
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    if os.path.exists(agent_main_path):
        with open(agent_main_path, 'r') as f:
            agent_content = f.read()
        
        # Check if there's an import for DhariwalUNet that's missing
        # Look for patterns like "from X import DhariwalUNet"
        import re
        imports = re.findall(r'from\s+(\S+)\s+import.*DhariwalUNet', agent_content)
        for imp_mod in imports:
            try:
                mod = __import__(imp_mod, fromlist=['DhariwalUNet'])
                cls = getattr(mod, 'DhariwalUNet', None)
                if cls is not None:
                    _dhariwal_unet_class = cls
                    import builtins
                    builtins.DhariwalUNet = cls
                    break
            except ImportError:
                continue

# If still not found, create a placeholder and patch agent_main source
if _dhariwal_unet_class is None:
    # We need to find the actual class. Let's check if there's a networks.py or similar
    # Let's try loading via dill from the pkl to see if it has the class
    try:
        data_path = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_diffpir_sandbox/run_code/std_data/data_main.pkl'
        with open(data_path, 'rb') as f:
            _temp_data = dill.load(f)
        # Check if output contains model with DhariwalUNet
        del _temp_data
    except Exception:
        pass

# Final attempt: look for the class in any .py file more aggressively
if _dhariwal_unet_class is None:
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
    ]
    # Also check if there's an InverseBench directory
    inversebench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'InverseBench')
    if os.path.exists(inversebench_dir):
        search_dirs.append(inversebench_dir)
        sys.path.insert(0, inversebench_dir)
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            # Skip hidden dirs and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for fname in files:
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r') as f:
                            content = f.read()
                        if 'class DhariwalUNet' in content:
                            # Add parent to sys.path
                            parent_dir = os.path.dirname(fpath)
                            if parent_dir not in sys.path:
                                sys.path.insert(0, parent_dir)
                            mod_name = fname.replace('.py', '')
                            mod = __import__(mod_name, fromlist=['DhariwalUNet'])
                            cls = getattr(mod, 'DhariwalUNet', None)
                            if cls is not None:
                                _dhariwal_unet_class = cls
                                import builtins
                                builtins.DhariwalUNet = cls
                                break
                    except Exception:
                        continue
            if _dhariwal_unet_class is not None:
                break

# If we absolutely cannot find it, we need to patch agent_main.py dynamically
if _dhariwal_unet_class is None:
    # Create a module-level patch: modify agent_main before importing
    import importlib
    import types
    
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    if os.path.exists(agent_main_path):
        with open(agent_main_path, 'r') as f:
            source = f.read()
        
        # Check if there's a run_code directory with relevant modules
        run_code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_code')
        if os.path.exists(run_code_dir):
            sys.path.insert(0, run_code_dir)
            # Search in run_code
            for root, dirs, files in os.walk(run_code_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for fname in files:
                    if fname.endswith('.py'):
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, 'r') as f:
                                content = f.read()
                            if 'class DhariwalUNet' in content:
                                parent_dir = os.path.dirname(fpath)
                                if parent_dir not in sys.path:
                                    sys.path.insert(0, parent_dir)
                                mod_name = fname.replace('.py', '')
                                mod = __import__(mod_name, fromlist=['DhariwalUNet'])
                                cls = getattr(mod, 'DhariwalUNet', None)
                                if cls is not None:
                                    _dhariwal_unet_class = cls
                                    import builtins
                                    builtins.DhariwalUNet = cls
                                    break
                        except Exception as e:
                            continue
                if _dhariwal_unet_class is not None:
                    break

# Ultra-last resort: try to find via pip-installed packages or site-packages
if _dhariwal_unet_class is None:
    try:
        # Maybe it's in a package called 'inversebench' or similar
        from inversebench.models import DhariwalUNet as _cls
        _dhariwal_unet_class = _cls
        import builtins
        builtins.DhariwalUNet = _cls
    except ImportError:
        pass

if _dhariwal_unet_class is None:
    try:
        from guided_diffusion.unet import UNetModel as DhariwalUNet
        _dhariwal_unet_class = DhariwalUNet
        import builtins
        builtins.DhariwalUNet = DhariwalUNet
    except ImportError:
        pass

# Now try to import agent_main with patching if needed
if _dhariwal_unet_class is None:
    # We'll need to dynamically modify agent_main.py
    # Read the source, add a stub DhariwalUNet class, and exec it
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    
    # Try to find DhariwalUNet by looking at all .py files in parent directories too
    for search_root in [os.path.dirname(os.path.abspath(__file__)), 
                         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'venv' and d != '.git']
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', errors='ignore') as f:
                        first_lines = f.read(50000)
                    if 'class DhariwalUNet' in first_lines:
                        parent = os.path.dirname(fpath)
                        if parent not in sys.path:
                            sys.path.insert(0, parent)
                        # Also try grandparent for package imports
                        grandparent = os.path.dirname(parent)
                        if grandparent not in sys.path:
                            sys.path.insert(0, grandparent)
                        mod_name = fname.replace('.py', '')
                        try:
                            mod = __import__(mod_name, fromlist=['DhariwalUNet'])
                            cls = getattr(mod, 'DhariwalUNet', None)
                            if cls is not None:
                                _dhariwal_unet_class = cls
                                import builtins
                                builtins.DhariwalUNet = cls
                                raise StopIteration
                        except StopIteration:
                            raise
                        except Exception:
                            pass
                except StopIteration:
                    break
                except Exception:
                    continue
            if _dhariwal_unet_class is not None:
                break
        if _dhariwal_unet_class is not None:
            break

# Now attempt to import agent_main
try:
    from agent_main import main
except NameError as e:
    if 'DhariwalUNet' in str(e):
        # Dynamically patch and load
        print(f"DhariwalUNet not found via normal imports. Attempting dynamic patch...")
        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        with open(agent_main_path, 'r') as f:
            source = f.read()
        
        # Try to create a minimal stub - but this won't work for actual model loading
        # Instead, let's see if the pkl data has the class embedded
        try:
            data_path = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_diffpir_sandbox/run_code/std_data/data_main.pkl'
            with open(data_path, 'rb') as f:
                _temp = dill.load(f)
            # dill may have registered the class
            # Try import again
            try:
                from agent_main import main
            except:
                raise
        except Exception:
            print(f"FATAL: Cannot resolve DhariwalUNet dependency")
            traceback.print_exc()
            sys.exit(1)

from verification_utils import recursive_check


def load_data(path):
    """Load a pickle data file using dill."""
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data


def test_main():
    """Test the main function against stored standard data."""
    
    # --- Data paths ---
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_diffpir_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # --- Determine scenario ---
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    assert outer_path is not None, f"Could not find outer data file (data_main.pkl) in: {data_paths}"
    
    # --- Phase 1: Load outer data ---
    print(f"Loading outer data from: {outer_path}")
    try:
        outer_data = load_data(outer_path)
    except Exception as e:
        print(f"FAILED to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    print(f"Outer data loaded. func_name={outer_data.get('func_name', 'N/A')}")
    print(f"  args count: {len(outer_args)}")
    print(f"  kwargs keys: {list(outer_kwargs.keys()) if outer_kwargs else '(none)'}")
    print(f"  expected output type: {type(expected_output)}")
    
    # --- Phase 2: Execute main ---
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B (Factory/Closure pattern)")
        print(f"Running main(*args, **kwargs) to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Verify operator is callable
        if not callable(agent_operator):
            print(f"WARNING: agent_operator is not callable (type={type(agent_operator)}), "
                  f"proceeding with inner data comparison anyway.")
        
        for inner_path in sorted(inner_paths):
            print(f"\nLoading inner data from: {inner_path}")
            try:
                inner_data = load_data(inner_path)
            except Exception as e:
                print(f"FAILED to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            print(f"Executing operator with inner args...")
            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAILED to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            print("Comparing inner results...")
            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAILED during comparison: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED for {os.path.basename(inner_path)}: {msg}")
                sys.exit(1)
            else:
                print(f"TEST PASSED for {os.path.basename(inner_path)}")
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A (Simple function call)")
        print(f"Running main(*args, **kwargs)...")
        try:
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAILED to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected_output)}")
        
        # --- Phase 3: Comparison ---
        print("Comparing results...")
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
            print("TEST PASSED")
    
    sys.exit(0)


if __name__ == '__main__':
    test_main()