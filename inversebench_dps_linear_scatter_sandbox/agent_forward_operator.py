import torch


# --- Extracted Dependencies ---

def full_propagate_to_sensor(f, utot_dom_set, sensor_greens_function_set, dx, dy):
    num_trans = utot_dom_set.shape[2]
    num_rec = sensor_greens_function_set.shape[2]
    contSrc = f[0, 0].unsqueeze(-1) * utot_dom_set
    conjSrc = torch.conj(contSrc).reshape(-1, num_trans)
    sensor_greens_func = sensor_greens_function_set.reshape(-1, num_rec)
    uscat_pred_set = dx * dy * torch.matmul(conjSrc.T, sensor_greens_func)
    return uscat_pred_set

def forward_operator(x: torch.Tensor, forward_op_params: dict, unnormalize: bool = True) -> torch.Tensor:
    """
    The forward operator that maps image x to measurements y_pred.
    Implements the scattering forward model.
    """
    dx = forward_op_params['dx']
    dy = forward_op_params['dy']
    unnorm_shift = forward_op_params['unnorm_shift']
    unnorm_scale = forward_op_params['unnorm_scale']
    sensor_greens_function_set = forward_op_params['sensor_greens_function_set']
    uinc_dom_set = forward_op_params['uinc_dom_set']
    
    f = x.to(torch.float64)
    if unnormalize:
        f = (f + unnorm_shift) * unnorm_scale
    
    uscat_pred_set = full_propagate_to_sensor(
        f, 
        uinc_dom_set[..., 0], 
        sensor_greens_function_set[..., 0], 
        dx, 
        dy
    )
    return uscat_pred_set.unsqueeze(0)
