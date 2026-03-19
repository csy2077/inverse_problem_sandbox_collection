import sys
import os
import dill
import torch
import numpy as np
import traceback
import torch.nn.functional as F


def patch_all_unet_block_references():
    """
    Find the actual UNetBlock class from gen_std_data and ensure ALL methods
    that reference 'UNetBlock' in their globals have the correct reference.
    """
    try:
        import gen_std_data
    except ImportError:
        return None

    # Ensure F is available in gen_std_data
    gen_std_data.F = F

    # Find UNetBlock class defined in gen_std_data
    unet_block_cls = None
    if hasattr(gen_std_data, 'UNetBlock') and isinstance(gen_std_data.UNetBlock, type):
        unet_block_cls = gen_std_data.UNetBlock

    if unet_block_cls is None:
        for name, obj in list(gen_std_data.__dict__.items()):
            if isinstance(obj, type) and 'UNetBlock' in name:
                unet_block_cls = obj
                break

    return unet_block_cls


def deep_patch_model(net):
    """
    After loading the model, find the ACTUAL UNetBlock class from the model's
    modules and patch every method's __globals__ that references UNetBlock.
    """
    if not isinstance(net, torch.nn.Module):
        return

    # Step 1: Find the actual UNetBlock class from model instances
    actual_unet_block = None
    all_modules = list(net.modules())
    if hasattr(net, 'model') and isinstance(net.model, torch.nn.Module):
        all_modules.extend(list(net.model.modules()))

    for child in all_modules:
        if type(child).__name__ == 'UNetBlock':
            actual_unet_block = type(child)
            break

    if actual_unet_block is None:
        print("WARNING: No UNetBlock instances found in model")
        return

    print(f"Found actual UNetBlock class: id={id(actual_unet_block)}")

    # Step 2: Patch ALL methods in ALL module classes to use the correct UNetBlock
    patched_classes = set()
    for child in all_modules:
        cls = type(child)
        cls_id = id(cls)
        if cls_id in patched_classes:
            continue
        patched_classes.add(cls_id)

        # Walk the MRO to patch all inherited methods too
        for klass in cls.__mro__:
            klass_id = id(klass)
            if klass_id in patched_classes:
                continue
            patched_classes.add(klass_id)

            for attr_name in list(klass.__dict__.keys()):
                try:
                    attr = klass.__dict__[attr_name]
                    # Handle regular functions/methods
                    func = None
                    if hasattr(attr, '__func__'):
                        func = attr.__func__
                    elif hasattr(attr, '__globals__'):
                        func = attr
                    elif hasattr(attr, 'fget') and hasattr(attr.fget, '__globals__'):
                        func = attr.fget

                    if func is not None and hasattr(func, '__globals__'):
                        needs_patch = False
                        if 'UNetBlock' in func.__globals__:
                            if func.__globals__['UNetBlock'] is not actual_unet_block:
                                func.__globals__['UNetBlock'] = actual_unet_block
                                needs_patch = True
                        # Also check if UNetBlock is referenced but not in globals
                        # (it would cause NameError)
                        if 'F' not in func.__globals__ or func.__globals__['F'] is not F:
                            func.__globals__['F'] = F
                except Exception:
                    pass

    # Step 3: Also directly patch the forward method of SongUNet-like class
    # by looking for the method that contains the isinstance check
    if hasattr(net, 'model'):
        model = net.model
        model_cls = type(model)
        if hasattr(model_cls, 'forward'):
            fwd = model_cls.forward
            if hasattr(fwd, '__globals__'):
                fwd.__globals__['UNetBlock'] = actual_unet_block
                fwd.__globals__['F'] = F
                print(f"  Patched {model_cls.__name__}.forward globals with UNetBlock id={id(actual_unet_block)}")

    # Step 4: Patch net's own __call__ / forward
    net_cls = type(net)
    for method_name in ['forward', '__call__']:
        if hasattr(net_cls, method_name):
            method = getattr(net_cls, method_name)
            if hasattr(method, '__globals__'):
                method.__globals__['UNetBlock'] = actual_unet_block
                method.__globals__['F'] = F


def inject_unet_block_into_globals_of_all_loaded_classes():
    """
    Nuclear option: scan ALL loaded modules for any class that has a forward
    method referencing UNetBlock and patch it.
    """
    import gc
    
    # Find all UNetBlock classes
    unet_classes = []
    actual_unet = None
    
    for obj in gc.get_objects():
        try:
            if isinstance(obj, type) and obj.__name__ == 'UNetBlock' and issubclass(obj, torch.nn.Module):
                unet_classes.append(obj)
        except Exception:
            pass

    # Find the one that actually has instances
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module) and type(obj).__name__ == 'UNetBlock':
                actual_unet = type(obj)
                break
        except Exception:
            pass

    if actual_unet is None and unet_classes:
        actual_unet = unet_classes[0]

    if actual_unet is None:
        return

    print(f"GC scan found actual UNetBlock: id={id(actual_unet)}")

    # Now find all functions that reference UNetBlock in their globals
    patched_count = 0
    for obj in gc.get_objects():
        try:
            if hasattr(obj, '__globals__') and callable(obj):
                g = obj.__globals__
                if 'UNetBlock' in g and g['UNetBlock'] is not actual_unet:
                    g['UNetBlock'] = actual_unet
                    g['F'] = F
                    patched_count += 1
        except Exception:
            pass

    print(f"  GC patched {patched_count} function globals")


# Pre-patch
initial_unet_block = patch_all_unet_block_references()

from agent_ode_sampler import ode_sampler
from verification_utils import recursive_check


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
        patch_all_unet_block_references()

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

    # Critical: Patch UNetBlock references AFTER loading the model
    if len(outer_args) > 0:
        net = outer_args[0]
        if isinstance(net, torch.nn.Module):
            print("Patching UNetBlock references in loaded model...")
            deep_patch_model(net)
            # Also do the nuclear GC-based patching
            inject_unet_block_into_globals_of_all_loaded_classes()

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