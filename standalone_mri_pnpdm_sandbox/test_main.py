import sys
import os
import dill
import torch
import numpy as np
import traceback

# We need to make DhariwalUNet available before importing agent_main
# First, try to find and import it from the environment
try:
    # Try importing from common locations
    try:
        from torch_utils import distributed as dist
    except ImportError:
        pass

    # Try to find DhariwalUNet in various possible modules
    DhariwalUNet = None

    # Attempt 1: from a networks module
    try:
        from networks import DhariwalUNet
    except ImportError:
        pass

    # Attempt 2: from edm or diffusion related modules
    if DhariwalUNet is None:
        try:
            from training.networks import DhariwalUNet
        except ImportError:
            pass

    # Attempt 3: from torch_utils
    if DhariwalUNet is None:
        try:
            from torch_utils.networks import DhariwalUNet
        except ImportError:
            pass

    # Attempt 4: search in sys.path for any module containing DhariwalUNet
    if DhariwalUNet is None:
        try:
            import importlib
            import pkgutil
            for finder, name, ispkg in pkgutil.iter_modules():
                try:
                    mod = importlib.import_module(name)
                    if hasattr(mod, 'DhariwalUNet'):
                        DhariwalUNet = getattr(mod, 'DhariwalUNet')
                        break
                except:
                    continue
        except:
            pass

    # Attempt 5: try edm_networks or similar
    if DhariwalUNet is None:
        try:
            from guided_diffusion.unet import UNetModel as DhariwalUNet
        except ImportError:
            pass

    # Attempt 6: look for it in inversebench or similar packages
    if DhariwalUNet is None:
        try:
            from inversebench.networks import DhariwalUNet
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from model import DhariwalUNet
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from models import DhariwalUNet
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from edm import DhariwalUNet
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from network import DhariwalUNet
        except ImportError:
            pass

    # Attempt: scan all .py files in current directory and parent directories
    if DhariwalUNet is None:
        search_dirs = ['.', '..', '../..']
        for sd in search_dirs:
            if not os.path.isdir(sd):
                continue
            for root, dirs, files in os.walk(sd):
                for fname in files:
                    if fname.endswith('.py') and fname not in ('test_main.py', 'agent_main.py'):
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, 'r') as f:
                                content = f.read()
                            if 'class DhariwalUNet' in content:
                                # Try to get module name
                                rel = os.path.relpath(fpath, '.')
                                mod_name = rel.replace('/', '.').replace('\\', '.').replace('.py', '')
                                try:
                                    mod = importlib.import_module(mod_name)
                                    if hasattr(mod, 'DhariwalUNet'):
                                        DhariwalUNet = getattr(mod, 'DhariwalUNet')
                                        break
                                except:
                                    # Try exec approach
                                    ns = {}
                                    exec(compile(content, fpath, 'exec'), ns)
                                    if 'DhariwalUNet' in ns:
                                        DhariwalUNet = ns['DhariwalUNet']
                                        break
                        except:
                            continue
                if DhariwalUNet is not None:
                    break
            if DhariwalUNet is not None:
                break

    # If found, inject into builtins so agent_main can find it
    if DhariwalUNet is not None:
        import builtins
        builtins.DhariwalUNet = DhariwalUNet
    else:
        # As a last resort, try loading from the pkl file to get the class
        try:
            outer_path_tmp = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_pnpdm_sandbox/run_code/std_data/data_main.pkl'
            with open(outer_path_tmp, 'rb') as f:
                tmp_data = dill.load(f)
            # The pkl might have serialized objects that reference DhariwalUNet
            # Check if we can extract it
            import builtins
            if not hasattr(builtins, 'DhariwalUNet'):
                # Create a dummy that will be replaced
                pass
        except:
            pass

except Exception as e:
    print(f"Warning during DhariwalUNet discovery: {e}")
    traceback.print_exc()

# Determine data paths and classify them
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_pnpdm_sandbox/run_code/std_data/data_main.pkl']

outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    else:
        outer_path = p


def _ensure_dhariwal_unet():
    """Make sure DhariwalUNet is available in the agent_main module's namespace."""
    import builtins
    if hasattr(builtins, 'DhariwalUNet'):
        return True

    # Try to find it by loading the pickle first - dill may restore it
    try:
        with open(outer_path, 'rb') as f:
            tmp = dill.load(f)
        # After loading, dill may have restored needed classes
        # Check if any loaded objects reference DhariwalUNet
        del tmp
    except:
        pass

    # Try to find in already loaded modules
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, 'DhariwalUNet'):
            builtins.DhariwalUNet = getattr(mod, 'DhariwalUNet')
            return True

    # Search more aggressively
    try:
        import importlib
        # Common EDM-style module paths
        candidates = [
            'edm.networks', 'edm.training.networks', 'training.networks',
            'networks', 'network', 'models.networks', 'diffusion.networks',
            'inversebench.model', 'inversebench.networks',
            'score_sde.models.ncsnpp', 'ncsnpp',
        ]
        for c in candidates:
            try:
                m = importlib.import_module(c)
                if hasattr(m, 'DhariwalUNet'):
                    builtins.DhariwalUNet = getattr(m, 'DhariwalUNet')
                    return True
            except:
                continue
    except:
        pass

    return False


def _find_and_inject_dhariwal():
    """
    Comprehensive search for DhariwalUNet class definition.
    This scans Python files to find and load the class.
    """
    import builtins
    if hasattr(builtins, 'DhariwalUNet'):
        return

    # Search in the sandbox directory and common locations
    search_roots = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_pnpdm_sandbox',
        '/fs-computility-new/UPDZ02_sunhe/shared/InverseBench',
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ]

    for search_root in search_roots:
        if not os.path.isdir(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            # Skip common non-relevant dirs
            skip = False
            for skip_dir in ['__pycache__', '.git', 'node_modules', 'egg-info']:
                if skip_dir in root:
                    skip = True
                    break
            if skip:
                continue

            for fname in files:
                if not fname.endswith('.py'):
                    continue
                if fname in ('test_main.py',):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', errors='ignore') as f:
                        content = f.read()
                    if 'class DhariwalUNet' not in content:
                        continue
                    print(f"Found DhariwalUNet in: {fpath}")

                    # Add the directory to sys.path
                    fdir = os.path.dirname(fpath)
                    if fdir not in sys.path:
                        sys.path.insert(0, fdir)

                    # Try importing as module
                    mod_name = fname.replace('.py', '')
                    try:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        if hasattr(mod, 'DhariwalUNet'):
                            builtins.DhariwalUNet = getattr(mod, 'DhariwalUNet')
                            print(f"Successfully loaded DhariwalUNet from module: {mod_name}")
                            return
                    except Exception as e2:
                        print(f"Module import failed for {mod_name}: {e2}")

                except Exception as e:
                    continue

    print("WARNING: Could not find DhariwalUNet class definition anywhere.")


def test_main():
    """Test the main function against captured standard data."""

    # First, ensure DhariwalUNet is available
    _find_and_inject_dhariwal()
    _ensure_dhariwal_unet()

    try:
        from agent_main import main
    except NameError as e:
        if 'DhariwalUNet' in str(e):
            print(f"FAIL: DhariwalUNet not found. Error: {e}")
            # Try one more approach: patch agent_main's globals
            try:
                import builtins
                # Create a minimal stub if truly not found anywhere
                # This is a last resort - we need to find the real class
                print("Attempting to find DhariwalUNet via dill-loaded objects...")

                # Load the pkl which might contain serialized model
                with open(outer_path, 'rb') as f:
                    outer_data = dill.load(f)

                # Check if the output contains the net with model attribute
                output = outer_data.get('output', None)
                if output is not None and hasattr(output, '__class__'):
                    print(f"Output class: {output.__class__}")

                # Try to get DhariwalUNet from dill's loaded namespace
                for mod_name, mod in list(sys.modules.items()):
                    if mod is not None and hasattr(mod, 'DhariwalUNet'):
                        builtins.DhariwalUNet = getattr(mod, 'DhariwalUNet')
                        print(f"Found DhariwalUNet in module: {mod_name}")
                        break

                from agent_main import main
            except Exception as e2:
                print(f"FAIL: Could not import agent_main even after retry: {e2}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"FAIL: Import error: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Could not import agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    from verification_utils import recursive_check

    assert outer_path is not None, f"Could not find outer data file in {data_paths}"
    assert os.path.exists(outer_path), f"Outer data file does not exist: {outer_path}"

    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    print(f"Number of args: {len(outer_args)}, Number of kwargs: {len(outer_kwargs)}")
    print(f"Expected output type: {type(expected_output)}")

    if len(inner_paths) == 0:
        # Scenario A: Simple function call
        print("Scenario A: Simple function - running main(*args, **kwargs)")
        try:
            result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: main() raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        print(f"Result type: {type(result)}")
        print(f"Expected type: {type(expected_output)}")

        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario B: Factory/Closure pattern
        print("Scenario B: Factory/Closure pattern detected")
        print("Phase 1: Reconstructing operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: main() raised an exception during operator creation: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable operator, got {type(agent_operator)}")
            sys.exit(1)

        print(f"Operator type: {type(agent_operator)}")

        # Phase 2: Execute with inner data
        for inner_path in sorted(inner_paths):
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)

            print(f"Inner data keys: {list(inner_data.keys())}")
            print(f"Inner function name: {inner_data.get('func_name', 'unknown')}")
            print(f"Number of inner args: {len(inner_args)}, Number of inner kwargs: {len(inner_kwargs)}")

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: Operator execution raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            print(f"Result type: {type(result)}")
            print(f"Expected type: {type(inner_expected)}")

            try:
                passed, msg = recursive_check(inner_expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for: {os.path.basename(inner_path)}")

        print("TEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    test_main()