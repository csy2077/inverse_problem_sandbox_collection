import torch


# --- Extracted Dependencies ---

def TV3D(arr):
    """
    Calculates total variation regularization of input 3D array,
    as the sum of total variation along each axis.
    """
    return torch.mean(
        torch.ravel(torch.abs(arr[1:, :, :] - arr[:-1, :, :])) +
        torch.ravel(torch.abs(arr[:, 1:, :] - arr[:, :-1, :])) +
        torch.ravel(torch.abs(arr[:, :, 1:] - arr[:, :, :-1]))
    )
