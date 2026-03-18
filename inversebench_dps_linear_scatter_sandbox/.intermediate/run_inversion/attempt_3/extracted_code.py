import sys
import os
import dill
import numpy as np
import traceback
import logging
import torch
import torch.nn.functional as F

# Import target function
from agent_run_inversion import run_inversion


# --- Inject Referee (Evaluation Logic) ---
def evaluate_results(recon: torch.Tensor, 
                     target: torch.Tensor, 
                     observation: torch.Tensor,
                     forward_op_params: dict,
                     logger: logging.Logger) -> dict:
    """
    Evaluates reconstruction quality using PSNR and SSIM metrics.
    """
    from piq import psnr, ssim
    
    unnorm_shift = forward_op_params['unnorm_shift']
    unnorm_scale = forward_op_params['unnorm_scale']
    
    # Unnormalize for evaluation
    recon_unnorm = (recon + unnorm_shift) * unnorm_scale
    target_unnorm = (target + unnorm_shift) * unnorm_scale
    
    # Clip to [0, 1] for metric computation
    recon_clipped = recon_unnorm.clip(0, 1)
    target_clipped = target_unnorm.clip(0, 1)
    
    # Ensure same shape for metrics
    if recon_clipped.shape != target_clipped.shape:
        target_clipped = target_clipped.repeat(recon_clipped.shape[0], 1, 1, 1)
    
    # Compute metrics
    psnr_val = psnr(recon_clipped, target_clipped, data_range=1.0, reduction='none').mean().item()
    ssim_val = ssim(recon_clipped, target_clipped, data_range=1.0, reduction='none').mean().item()
    
    metric_dict = {
        'psnr': psnr_val,
        'ssim': ssim_val
    }
    
    logger.info(f"Metric results: {metric_dict}")
    
    return metric_dict


def setup_logger():
    """Setup a simple logger for evaluation."""
    logger = logging.getLogger('test_run_inversion')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def patch_unetblock_in_module(module_obj):
    """
    Find UNetBlock class and inject it into gen_std_data module globals.
    """
    import gen_std_data
    
    # First, find UNetBlock class from the module's own definitions
    unetblock_class = None
    
    # Search through all classes defined in gen_std_data
    for name in dir(gen_std_data):
        obj = getattr(gen_std_data, name, None)
        if isinstance(obj, type) and 'UNetBlock' in name:
            unetblock_class = obj
            break
    
    # If not found at module level, search through loaded model
    if unetblock_class is None and module_obj is not None:
        for submodule in module_obj.modules():
            cls = submodule.__class__
            if hasattr(cls, '__globals__') and 'UNetBlock' in cls.__globals__:
                unetblock_class = cls.__globals__['UNetBlock']
                break
    
    # Now inject UNetBlock into all forward method globals
    if module_obj is not None:
        for submodule in module_obj.modules():
            cls = submodule.__class__
            
            # Inject into __init__ method globals
            if hasattr(cls, '__init__'):
                init_method = cls.__init__
                if hasattr(init_method, '__globals__'):
                    init_method.__globals__['F'] = F
                    if unetblock_class is not None:
                        init_method.__globals__['UNetBlock'] = unetblock_class
            
            # Inject into forward method globals
            if hasattr(submodule, 'forward'):
                forward = submodule.forward
                if hasattr(forward, '__func__'):
                    func = forward.__func__
                else:
                    func = forward
                if hasattr(func, '__globals__'):
                    func.__globals__['F'] = F
                    if unetblock_class is not None:
                        func.__globals__['UNetBlock'] = unetblock_class
    
    # Also set at module level
    if unetblock_class is not None:
        gen_std_data.UNetBlock = unetblock_class
    gen_std_data.F = F
    
    return unetblock_class


def find_and_inject_unetblock(data_dict):
    """
    Find UNetBlock from loaded data and inject it everywhere needed.
    """
    import gen_std_data
    
    unetblock_class = None
    
    # First try to find UNetBlock from the model in kwargs
    net = data_dict.get('kwargs', {}).get('net', None)
    
    if net is not None:
        # Search through all modules
        for submodule in net.modules():
            cls = submodule.__class__
            if hasattr(cls, '__globals__'):
                if 'UNetBlock' in cls.__globals__:
                    unetblock_class = cls.__globals__['UNetBlock']
                    break
    
    # If still not found, try other methods
    if unetblock_class is None:
        # Check if there's a class in gen_std_data with UNetBlock in name
        for name in dir(gen_std_data):
            obj = getattr(gen_std_data, name, None)
            if isinstance(obj, type) and 'UNetBlock' in name:
                unetblock_class = obj
                break
    
    # Now inject everywhere
    if unetblock_class is not None:
        gen_std_data.UNetBlock = unetblock_class
    gen_std_data.F = F
    
    # Inject into all model methods
    if net is not None:
        for submodule in net.modules():
            cls = submodule.__class__
            
            # Get all methods of the class
            for attr_name in dir(cls):
                attr = getattr(cls, attr_name, None)
                if callable(attr) and hasattr(attr, '__globals__'):
                    attr.__globals__['F'] = F
                    if unetblock_class is not None:
                        attr.__globals__['UNetBlock'] = unetblock_class
            
            # Also check instance methods
            if hasattr(submodule, 'forward'):
                forward = submodule.forward
                if hasattr(forward, '__func__'):
                    func = forward.__func__
                elif hasattr(forward, '__globals__'):
                    func = forward
                else:
                    continue
                    
                if hasattr(func, '__globals__'):
                    func.__globals__['F'] = F
                    if unetblock_class is not None:
                        func.__globals__['UNetBlock'] = unetblock_class
    
    return unetblock_class


def comprehensive_patch(net):
    """
    Comprehensively patch all modules to have UNetBlock and F available.
    """
    import gen_std_data
    
    # First pass: find UNetBlock
    unetblock_class = None
    all_globals = []
    
    for submodule in net.modules():
        cls = submodule.__class__
        
        # Collect all globals dicts
        for attr_name in ['forward', '__init__', '__call__']:
            if hasattr(cls, attr_name):
                method = getattr(cls, attr_name)
                if hasattr(method, '__globals__'):
                    all_globals.append(method.__globals__)
                    if 'UNetBlock' in method.__globals__:
                        unetblock_class = method.__globals__['UNetBlock']
        
        # Also check instance methods
        if hasattr(submodule, 'forward'):
            forward = submodule.forward
            if hasattr(forward, '__func__'):
                if hasattr(forward.__func__, '__globals__'):
                    all_globals.append(forward.__func__.__globals__)
                    if 'UNetBlock' in forward.__func__.__globals__:
                        unetblock_class = forward.__func__.__globals__['UNetBlock']
    
    # Second pass: inject into all globals
    for globals_dict in all_globals:
        globals_dict['F'] = F
        if unetblock_class is not None:
            globals_dict['UNetBlock'] = unetblock_class
    
    # Also set at module level
    gen_std_data.F = F
    if unetblock_class is not None:
        gen_std_data.UNetBlock = unetblock_class
    
    return unetblock_class


def main():
    logger = setup_logger()
    
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_run_inversion.pkl']
    
    # Analyze data files
    outer_data_path = None
    inner_data_paths = []
    
    for path in data_paths:
        filename = os.path.basename(path)
        if 'parent_function' in filename:
            inner_data_paths.append(path)
        else:
            outer_data_path = path
    
    if outer_data_path is None:
        print("ERROR: Could not find primary data file.")
        sys.exit(1)
    
    try:
        # Load outer/primary data
        print(f"Loading primary data from: {outer_data_path}")
        with open(outer_data_path, 'rb') as f:
            outer_data = dill.load(f)
        
        # Extract inputs and expected output
        args = outer_data.get('args', ())
        kwargs = outer_data.get('kwargs', {})
        std_output = outer_data.get('output', None)
        
        # Get the network and patch it comprehensively
        net = kwargs.get('net', None)
        if net is not None:
            print("Patching network with UNetBlock and F...")
            unetblock = comprehensive_patch(net)
            print(f"UNetBlock found and injected: {unetblock is not None}")
        
        # Additional data that might be needed for evaluation
        target = outer_data.get('target', None)
        observation = outer_data.get('observation', None)
        forward_op_params = outer_data.get('forward_op_params', None)
        
        # If forward_op_params not in outer_data, check kwargs
        if forward_op_params is None:
            forward_op_params = kwargs.get('forward_op_params', None)
        
        # If observation not in outer_data, check args/kwargs
        if observation is None:
            if len(args) > 0:
                observation = args[0]
            else:
                observation = kwargs.get('observation', None)
        
        print(f"Args count: {len(args)}")
        print(f"Kwargs keys: {list(kwargs.keys())}")
        
        # Check if this is chained execution
        if len(inner_data_paths) > 0:
            print("Detected chained execution pattern (Closure/Factory).")
            
            # Run outer function to get operator
            print("Running run_inversion to get operator...")
            operator = run_inversion(*args, **kwargs)
            
            # Load inner data and execute
            for inner_path in inner_data_paths:
                print(f"Loading inner data from: {inner_path}")
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                
                inner_args = inner_data.get('args', ())
                inner_kwargs = inner_data.get('kwargs', {})
                std_result = inner_data.get('output', None)
                
                # Update evaluation params from inner data if available
                if inner_data.get('target') is not None:
                    target = inner_data['target']
                if inner_data.get('forward_op_params') is not None:
                    forward_op_params = inner_data['forward_op_params']
                if inner_data.get('observation') is not None:
                    observation = inner_data['observation']
                
                print("Running operator with inner data...")
                agent_result = operator(*inner_args, **inner_kwargs)
        else:
            print("Detected direct execution pattern.")
            
            # Run the function directly
            print("Running run_inversion...")
            agent_result = run_inversion(*args, **kwargs)
            std_result = std_output
        
        print(f"Agent result shape: {agent_result.shape if hasattr(agent_result, 'shape') else type(agent_result)}")
        print(f"Standard result shape: {std_result.shape if hasattr(std_result, 'shape') else type(std_result)}")
        
        # Ensure tensors are on the same device
        if torch.is_tensor(agent_result) and torch.is_tensor(std_result):
            device = agent_result.device
            std_result = std_result.to(device)
            if target is not None and torch.is_tensor(target):
                target = target.to(device)
            if observation is not None and torch.is_tensor(observation):
                observation = observation.to(device)
            
            # Move forward_op_params tensors to device
            if forward_op_params is not None:
                for key, val in forward_op_params.items():
                    if torch.is_tensor(val):
                        forward_op_params[key] = val.to(device)
        
        # If target is not available, use std_result as target for comparison
        if target is None:
            print("Warning: No target found, using standard result as target for evaluation.")
            target = std_result
        
        # Evaluate both results
        print("\nEvaluating agent result...")
        score_agent = evaluate_results(agent_result, target, observation, forward_op_params, logger)
        
        print("\nEvaluating standard result...")
        score_std = evaluate_results(std_result, target, observation, forward_op_params, logger)
        
        # Extract primary metrics
        agent_psnr = score_agent['psnr']
        agent_ssim = score_agent['ssim']
        std_psnr = score_std['psnr']
        std_ssim = score_std['ssim']
        
        print(f"\n{'='*60}")
        print(f"Scores -> Agent: PSNR={agent_psnr:.4f}, SSIM={agent_ssim:.4f}")
        print(f"Scores -> Standard: PSNR={std_psnr:.4f}, SSIM={std_ssim:.4f}")
        print(f"{'='*60}")
        
        # Verification: Higher PSNR and SSIM are better
        # Allow 10% margin of error
        margin = 0.90
        
        psnr_pass = agent_psnr >= std_psnr * margin
        ssim_pass = agent_ssim >= std_ssim * margin
        
        print(f"\nPSNR Check: {'PASS' if psnr_pass else 'FAIL'} (Agent: {agent_psnr:.4f} vs Threshold: {std_psnr * margin:.4f})")
        print(f"SSIM Check: {'PASS' if ssim_pass else 'FAIL'} (Agent: {agent_ssim:.4f} vs Threshold: {std_ssim * margin:.4f})")
        
        if psnr_pass and ssim_pass:
            print("\n✓ Performance verification PASSED!")
            sys.exit(0)
        else:
            print("\n✗ Performance verification FAILED!")
            print("Agent performance degraded significantly compared to standard.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR during test execution:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()