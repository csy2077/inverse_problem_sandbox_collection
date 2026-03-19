import sys
import os
import dill
import torch
import numpy as np
import traceback
import torch.nn.functional as F

# Patch F into the gen_std_data module if it exists
try:
    import gen_std_data
    if not hasattr(gen_std_data, 'F'):
        gen_std_data.F = F
except ImportError:
    pass

# Also try to patch it into any module that might need it
import builtins
original_import = builtins.__import__

def patched_import(name, *args, **kwargs):
    mod = original_import(name, *args, **kwargs)
    if name == 'gen_std_data' or (hasattr(mod, '__file__') and mod.__file__ and 'gen_std_data' in str(mod.__file__)):
        if not hasattr(mod, 'F'):
            mod.F = F
    return mod

builtins.__import__ = patched_import

# Import the target function
from agent_ode_sampler import ode_sampler
from verification_utils import recursive_check


def ensure_F_available():
    """Ensure torch.nn.functional is available as F in all relevant modules."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, '__dict__'):
            if 'gen_std_data' in mod_name:
                if 'F' not in mod.__dict__:
                    mod.__dict__['F'] = F


def main():
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_navier_stokes_enkg_sandbox/run_code/std_data/data_ode_sampler.pkl'
    ]

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("FAIL: No outer data file found.")
        sys.exit(1)

    # Phase 1: Load outer data
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"  func_name: {outer_data.get('func_name', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Could not load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Ensure F is patched into all loaded modules after dill.load
    ensure_F_available()

    # Also patch F into the net object's module if needed
    if len(outer_args) > 0:
        net = outer_args[0]
        # Try to patch F into the module where net's class is defined
        if hasattr(net, '__class__') and hasattr(net.__class__, '__module__'):
            mod_name = net.__class__.__module__
            if mod_name in sys.modules and sys.modules[mod_name] is not None:
                if not hasattr(sys.modules[mod_name], 'F'):
                    sys.modules[mod_name].F = F
                    print(f"Patched F into module: {mod_name}")
        
        # Also check for model attribute and patch its module
        if hasattr(net, 'model'):
            model = net.model
            if hasattr(model, '__class__') and hasattr(model.__class__, '__module__'):
                mod_name = model.__class__.__module__
                if mod_name in sys.modules and sys.modules[mod_name] is not None:
                    if not hasattr(sys.modules[mod_name], 'F'):
                        sys.modules[mod_name].F = F
                        print(f"Patched F into model module: {mod_name}")

        # Brute force: patch F into ALL loaded modules that look relevant
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, '__dict__'):
                try:
                    if 'F' not in mod.__dict__:
                        # Check if this module has any reference to silu or similar
                        source_keys = list(mod.__dict__.keys())
                        for key in source_keys:
                            obj = mod.__dict__.get(key)
                            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                                mod.__dict__['F'] = F
                                break
                except Exception:
                    pass

    # Final ensure
    ensure_F_available()
    
    # Patch into __main__ and globals just in case
    try:
        import __main__
        __main__.F = F
    except:
        pass

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        try:
            agent_operator = ode_sampler(*outer_args, **outer_kwargs)
            print("Successfully created agent_operator from ode_sampler.")
        except Exception as e:
            print(f"FAIL: Error calling ode_sampler to create operator: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: agent_operator is not callable, got type: {type(agent_operator)}")
            sys.exit(1)

        for inner_path in inner_paths:
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data from: {inner_path}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
                print("Successfully executed agent_operator with inner args.")
            except Exception as e:
                print(f"FAIL: Error executing agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
                if not passed:
                    print(f"FAIL: Verification failed for {inner_path}")
                    print(f"  Message: {msg}")
                    sys.exit(1)
                else:
                    print(f"PASS: Verification succeeded for {inner_path}")
            except Exception as e:
                print(f"FAIL: Error during verification: {e}")
                traceback.print_exc()
                sys.exit(1)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            result = ode_sampler(*outer_args, **outer_kwargs)
            print("Successfully executed ode_sampler.")
        except Exception as e:
            print(f"FAIL: Error calling ode_sampler: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        try:
            passed, msg = recursive_check(expected, result)
            if not passed:
                print(f"FAIL: Verification failed.")
                print(f"  Message: {msg}")
                sys.exit(1)
            else:
                print("PASS: Verification succeeded.")
        except Exception as e:
            print(f"FAIL: Error during verification: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("TEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()