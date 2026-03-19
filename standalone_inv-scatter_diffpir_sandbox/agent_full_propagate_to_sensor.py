import torch


# --- Extracted Dependencies ---

def full_propagate_to_sensor(f, utot_dom_set, sensor_greens_function_set, dx, dy):
    """
    Propagate all the total fields to the sensors.

    Parameters:
    - f: (Ny x Nx) scattering potential
    - utot_dom_set: (Ny x Nx x numTrans) total field inside the computational domain
    - sensor_greens_function_set: (Ny x Nx x numRec) Green's functions
    - dx, dy: sampling steps

    Returns:
    - uscat_pred_set: (numTrans x numRec) predicted scattered fields
    """
    num_trans = utot_dom_set.shape[2]
    num_rec = sensor_greens_function_set.shape[2]
    contSrc = f[0, 0].unsqueeze(-1) * utot_dom_set    # (Ny x Nx x numTrans)
    conjSrc = torch.conj(contSrc).reshape(-1, num_trans)    # (Ny x Nx, numTrans)
    sensor_greens_func = sensor_greens_function_set.reshape(-1, num_rec)    # (Ny x Nx, numRec)
    uscat_pred_set = dx * dy * torch.matmul(conjSrc.T, sensor_greens_func)    # (numTrans, numRec)
    return uscat_pred_set
