import sys
import os
import dill
import torch
import numpy as np
import traceback
import torch.nn.functional as F

# We need to patch UNetBlock isinstance checks BEFORE loading pickled data.
# The core issue is that `isinstance(block, UNetBlock)` fails because the
# UNetBlock class reference in the forward method's globals doesn't match
# the actual class of the block objects. This causes blocks that ARE UNetBlocks
# to be called without the `emb` argument.

# Strategy: We need to find the UNetBlock class and ensure that the isinstance
# check in the model's forward method works correctly.

def patch_gen_std_data():
    """Patch gen_std_data module to fix UNetBlock isinstance checks."""
    try:
        import gen_std_data
    except ImportError:
        return None

    # Ensure F is available
    gen_std_data.F = F

    # Find UNetBlock class
    unet_block_class = None
    if hasattr(gen_std_data, 'UNetBlock') and isinstance(gen_std_data.UNetBlock, type):
        unet_block_class = gen_std_data.UNetBlock

    if unet_block_class is None:
        for name, obj in list(gen_std_data.__dict__.items()):
            if isinstance(obj, type) and 'UNetBlock' in name:
                unet_block_class = obj
                break

    # If still not found, try to extract from source
    if unet_block_class is None:
        try:
            source_file = gen_std_data.__file__
            if source_file and os.path.exists(source_file):
                with open(source_file, 'r') as f:
                    source_code = f.read()
                if 'class UNetBlock' in source_code:
                    ns = dict(gen_std_data.__dict__)
                    ns['F'] = F
                    ns['torch'] = torch
                    ns['np'] = np
                    exec(compile(source_code, source_file, 'exec'), ns)
                    if 'UNetBlock' in ns and isinstance(ns['UNetBlock'], type):
                        unet_block_class = ns['UNetBlock']
        except Exception as e:
            print(f"Warning: Source parsing failed: {e}")

    if unet_block_class is not None:
        gen_std_data.UNetBlock = unet_block_class
        # Patch all classes' methods
        for name, obj in list(gen_std_data.__dict__.items()):
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                for method_name in dir(obj):
                    try:
                        method = getattr(obj, method_name)
                        if callable(method) and hasattr(method, '__globals__'):
                            method.__globals__['F'] = F
                            method.__globals__['UNetBlock'] = unet_block_class
                    except Exception:
                        pass

    return unet_block_class


def deep_patch_module(module, unet_block_cls):
    """Recursively patch all nn.Module instances to have correct UNetBlock reference."""
    if module is None or unet_block_cls is None:
        return

    if not isinstance(module, torch.nn.Module):
        return

    patched_classes = set()

    for name, child in module.named_modules():
        child_class = type(child)
        class_id = id(child_class)
        if class_id in patched_classes:
            continue
        patched_classes.add(class_id)

        # Patch all methods of this class
        for method_name in dir(child_class):
            try:
                method = getattr(child_class, method_name)
                if callable(method) and hasattr(method, '__globals__'):
                    method.__globals__['F'] = F
                    method.__globals__['UNetBlock'] = unet_block_cls
            except Exception:
                pass


def patch_isinstance_in_forward(net, unet_block_cls):
    """
    The nuclear option: monkey-patch the forward method of the model that
    contains the isinstance(block, UNetBlock) check to use the correct class.
    
    We find the SongUNet-like model and replace its forward to always pass emb
    to UNetBlock instances (identified by class name rather than isinstance).
    """
    if net is None or not isinstance(net, torch.nn.Module):
        return

    # Find the inner model (the one with enc/dec blocks)
    target_model = None
    
    # Check if net has a .model attribute
    if hasattr(net, 'model') and isinstance(net.model, torch.nn.Module):
        target_model = net.model
    else:
        target_model = net

    # Now patch the forward of target_model to use class name check instead of isinstance
    original_forward = target_model.__class__.forward

    def patched_forward(self, *args, **kwargs):
        # We need to intercept and fix the isinstance check
        # Instead of patching forward entirely, let's just make sure UNetBlock is correct
        # in the forward's globals
        if hasattr(original_forward, '__globals__'):
            original_forward.__globals__['F'] = F
            if unet_block_cls is not None:
                original_forward.__globals__['UNetBlock'] = unet_block_cls
        return original_forward(self, *args, **kwargs)

    # Don't use this approach - instead ensure the globals are correct
    if hasattr(original_forward, '__globals__'):
        original_forward.__globals__['F'] = F
        if unet_block_cls is not None:
            original_forward.__globals__['UNetBlock'] = unet_block_cls


def find_real_unet_block_class(net):
    """Find the actual UNetBlock class by inspecting the model's children."""
    if not isinstance(net, torch.nn.Module):
        return None

    # Walk through all modules
    for name, child in net.named_modules():
        class_name = type(child).__name__
        if class_name == 'UNetBlock':
            return type(child)

    # Check net.model if exists
    if hasattr(net, 'model') and isinstance(net.model, torch.nn.Module):
        for name, child in net.model.named_modules():
            class_name = type(child).__name__
            if class_name == 'UNetBlock':
                return type(child)

    return None


def ensure_isinstance_works(net, unet_block_cls):
    """
    The isinstance check `isinstance(block, UNetBlock)` in the forward method
    must use the SAME class object as the actual class of the block instances.
    
    This function finds the actual class of UNetBlock instances in the model
    and ensures the forward method's globals point to that exact class.
    """
    if net is None or not isinstance(net, torch.nn.Module):
        return

    # Find the actual UNetBlock class from the model's children
    actual_unet_block = find_real_unet_block_class(net)
    if actual_unet_block is None:
        print("WARNING: Could not find any UNetBlock instances in the model")
        return

    print(f"Found actual UNetBlock class: {actual_unet_block}, id={id(actual_unet_block)}")

    if unet_block_cls is not None:
        print(f"Gen_std_data UNetBlock class: {unet_block_cls}, id={id(unet_block_cls)}")
        print(f"Classes match: {actual_unet_block is unet_block_cls}")

    # Now patch ALL forward methods to use the actual class
    patched = set()
    for name, child in net.named_modules():
        cls = type(child)
        if id(cls) in patched:
            continue
        patched.add(id(cls))

        for method_name in dir(cls):
            try:
                method = getattr(cls, method_name)
                if callable(method) and hasattr(method, '__globals__'):
                    if 'UNetBlock' in method.__globals__:
                        old_ref = method.__globals__['UNetBlock']
                        method.__globals__['UNetBlock'] = actual_unet_block
                        method.__globals__['F'] = F
                        if old_ref is not actual_unet_block:
                            print(f"  Patched {cls.__name__}.{method_name}: UNetBlock id {id(old_ref)} -> {id(actual_unet_block)}")
            except Exception:
                pass

    # Also patch net.model if it exists
    if hasattr(net, 'model') and isinstance(net.model, torch.nn.Module):
        for name, child in net.model.named_modules():
            cls = type(child)
            if id(cls) in patched:
                continue
            patched.add(id(cls))
            for method_name in dir(cls):
                try:
                    method = getattr(cls, method_name)
                    if callable(method) and hasattr(method, '__globals__'):
                        if 'UNetBlock' in method.__globals__:
                            old_ref = method.__globals__['UNetBlock']
                            method.__globals__['UNetBlock'] = actual_unet_block
                            method.__globals__['F'] = F
                except Exception:
                    pass


# Pre-patch gen_std_data
initial_unet_block = patch_gen_std_data()

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
        # Re-patch before loading
        patch_gen_std_data()

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

    # Critical: Find the ACTUAL UNetBlock class from the loaded model
    # and patch all forward methods to use it
    if len(outer_args) > 0:
        net = outer_args[0]
        if isinstance(net, torch.nn.Module):
            print("Patching UNetBlock references in loaded model...")
            ensure_isinstance_works(net, initial_unet_block)

            # Move model to appropriate device if needed
            # Check if x_initial has a device
            if len(outer_args) > 1 and isinstance(outer_args[1], torch.Tensor):
                device = outer_args[1].device
                print(f"  x_initial device: {device}")

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