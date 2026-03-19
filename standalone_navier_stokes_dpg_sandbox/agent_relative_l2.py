import torch


# --- Extracted Dependencies ---

def relative_l2(pred, target):
    '''
    Args:
        - pred (torch.Tensor): (N, C, H, W)
        - target (torch.Tensor): (C, H, W)
    Returns:
        - rel_l2 (torch.Tensor): (N,), relative L2 error
    '''
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2
