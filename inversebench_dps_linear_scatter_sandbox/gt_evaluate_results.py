import logging
import torch
from piq import psnr, ssim, SSIMLoss


# --- Extracted Dependencies ---

def evaluate_results(recon: torch.Tensor, 
                     target: torch.Tensor, 
                     observation: torch.Tensor,
                     forward_op_params: dict,
                     logger: logging.Logger) -> dict:
    """
    Evaluates reconstruction quality using PSNR and SSIM metrics.
    """
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
