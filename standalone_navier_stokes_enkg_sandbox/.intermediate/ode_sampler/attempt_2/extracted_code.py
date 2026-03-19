import sys
import os
import dill
import torch
import numpy as np
import traceback
import torch.nn.functional as F

# Pre-patch: We need to ensure UNetBlock and other names are available in gen_std_data
# before any deserialization or function calls happen
try:
    import gen_std_data
    # Patch F
    if not hasattr(gen_std_data, 'F'):
        gen_std_data.F = F
    
    # Find UNetBlock - it might be defined in gen_std_data but not at module scope
    # Search through all classes defined in gen_std_data
    unet_block_found = False
    for name, obj in list(gen_std_data.__dict__.items()):
        if name == 'UNetBlock':
            unet_block_found = True
            break
    
    if not unet_block_found:
        # Search for UNetBlock in the module's classes
        for name, obj in list(gen_std_data.__dict__.items()):
            if isinstance(obj, type) and 'UNetBlock' in name:
                gen_std_data.UNetBlock = obj
                unet_block_found = True
                break
    
    if not unet_block_found:
        # Try to find it by inspecting the source or looking for it in submodules
        # It might be a class that needs to be imported
        try:
            # Read the source file to find UNetBlock class definition
            source_file = gen_std_data.__file__
            if source_file:
                with open(source_file, 'r') as f:
                    source = f.read()
                
                # Check if UNetBlock is defined in the file
                if 'class UNetBlock' in source:
                    # It's defined but maybe not exported to module scope
                    # We need to exec the class definition in the module's namespace
                    import re
                    # Find the class and everything it needs
                    # A simpler approach: just re-execute the entire module with proper globals
                    exec(compile(source, source_file, 'exec'), gen_std_data.__dict__)
                    if hasattr(gen_std_data, 'UNetBlock'):
                        unet_block_found = True
        except Exception as e:
            print(f"Warning: Error trying to find UNetBlock from source: {e}")
    
    if not unet_block_found:
        # Last resort: look through all loaded modules
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, 'UNetBlock'):
                gen_std_data.UNetBlock = mod.UNetBlock
                unet_block_found = True
                break
    
    # Also ensure F is in the module's global namespace for forward methods
    # Patch the forward methods' globals if possible
    for name, obj in list(gen_std_data.__dict__.items()):
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
            if hasattr(obj, 'forward'):
                fwd = obj.forward
                if hasattr(fwd, '__globals__'):
                    if 'F' not in fwd.__globals__:
                        fwd.__globals__['F'] = F
                    if 'UNetBlock' not in fwd.__globals__ and unet_block_found:
                        fwd.__globals__['UNetBlock'] = gen_std_data.UNetBlock

except ImportError:
    pass

from agent_ode_sampler import ode_sampler
from verification_utils import recursive_check


def patch_all_modules():
    """Ensure F and UNetBlock are available in all relevant modules."""
    unet_block_class = None
    
    # First find UNetBlock
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, 'UNetBlock'):
            unet_block_class = mod.UNetBlock
            break
    
    # If still not found, search through all classes in gen_std_data
    if unet_block_class is None:
        try:
            import gen_std_data
            for name, obj in list(gen_std_data.__dict__.items()):
                if isinstance(obj, type) and 'UNet' in name and 'Block' in name:
                    unet_block_class = obj
                    gen_std_data.UNetBlock = obj
                    break
        except:
            pass
    
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not hasattr(mod, '__dict__'):
            continue
        try:
            if 'gen_std_data' in mod_name or mod_name == '__main__':
                if 'F' not in mod.__dict__:
                    mod.__dict__['F'] = F
                if unet_block_class is not None and 'UNetBlock' not in mod.__dict__:
                    mod.__dict__['UNetBlock'] = unet_block_class
        except Exception:
            pass
    
    return unet_block_class


def patch_net_object(net, unet_block_class):
    """Patch the net object and all its submodules to have F and UNetBlock in their forward.__globals__."""
    if net is None:
        return
    
    # Get the class of net
    net_class = type(net)
    
    # Patch forward method globals
    if hasattr(net_class, 'forward') and hasattr(net_class.forward, '__globals__'):
        net_class.forward.__globals__['F'] = F
        if unet_block_class is not None:
            net_class.forward.__globals__['UNetBlock'] = unet_block_class
    
    # Patch the module where the class is defined
    if hasattr(net_class, '__module__') and net_class.__module__ in sys.modules:
        mod = sys.modules[net_class.__module__]
        if mod is not None:
            if not hasattr(mod, 'F'):
                mod.F = F
            if unet_block_class is not None and not hasattr(mod, 'UNetBlock'):
                mod.UNetBlock = unet_block_class
    
    # Recursively patch submodules if it's an nn.Module
    if isinstance(net, torch.nn.Module):
        for child_name, child in net.named_modules():
            child_class = type(child)
            if hasattr(child_class, 'forward') and hasattr(child_class.forward, '__globals__'):
                child_class.forward.__globals__['F'] = F
                if unet_block_class is not None:
                    child_class.forward.__globals__['UNetBlock'] = unet_block_class
            
            if hasattr(child_class, '__module__') and child_class.__module__ in sys.modules:
                mod = sys.modules[child_class.__module__]
                if mod is not None:
                    if not hasattr(mod, 'F'):
                        mod.F = F
                    if unet_block_class is not None and not hasattr(mod, 'UNetBlock'):
                        mod.UNetBlock = unet_block_class


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

    # Patch all modules after loading
    unet_block_class = patch_all_modules()

    # If UNetBlock still not found, try harder
    if unet_block_class is None:
        try:
            import gen_std_data
            source_file = gen_std_data.__file__
            if source_file and os.path.exists(source_file):
                with open(source_file, 'r') as f:
                    source_code = f.read()
                
                if 'class UNetBlock' in source_code:
                    # Create a temporary namespace and exec to extract UNetBlock
                    temp_ns = {'torch': torch, 'F': F, 'np': np}
                    temp_ns.update(gen_std_data.__dict__)
                    
                    # Extract just the UNetBlock class definition and its dependencies
                    import re
                    # Find class definition
                    lines = source_code.split('\n')
                    in_class = False
                    class_lines = []
                    indent_level = None
                    
                    for line in lines:
                        if 'class UNetBlock' in line:
                            in_class = True
                            indent_level = len(line) - len(line.lstrip())
                            class_lines.append(line)
                            continue
                        if in_class:
                            if line.strip() == '':
                                class_lines.append(line)
                                continue
                            current_indent = len(line) - len(line.lstrip())
                            if current_indent > indent_level or line.strip() == '':
                                class_lines.append(line)
                            else:
                                break
                    
                    if class_lines:
                        class_code = '\n'.join(class_lines)
                        try:
                            exec(class_code, temp_ns)
                            if 'UNetBlock' in temp_ns:
                                unet_block_class = temp_ns['UNetBlock']
                                gen_std_data.UNetBlock = unet_block_class
                                print("Successfully extracted UNetBlock from source.")
                        except Exception as e2:
                            print(f"Warning: Could not exec UNetBlock class: {e2}")
        except Exception as e:
            print(f"Warning: Error in UNetBlock extraction: {e}")

    # Patch again after potential extraction
    if unet_block_class is not None:
        patch_all_modules()

    # Patch the net object specifically
    if len(outer_args) > 0:
        net = outer_args[0]
        patch_net_object(net, unet_block_class)
        
        # Also patch model inside net if it exists
        if hasattr(net, 'model'):
            patch_net_object(net.model, unet_block_class)

    # Final comprehensive patch: go through EVERY class in every module and patch forward globals
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not hasattr(mod, '__dict__'):
            continue
        try:
            for attr_name, attr_val in list(mod.__dict__.items()):
                if isinstance(attr_val, type) and issubclass(attr_val, torch.nn.Module):
                    if hasattr(attr_val, 'forward') and hasattr(attr_val.forward, '__globals__'):
                        attr_val.forward.__globals__['F'] = F
                        if unet_block_class is not None:
                            attr_val.forward.__globals__['UNetBlock'] = unet_block_class
        except Exception:
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