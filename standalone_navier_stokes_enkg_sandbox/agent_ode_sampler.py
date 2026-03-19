import torch


# --- Extracted Dependencies ---

def ode_sampler(
    net,
    x_initial,
    num_steps=18,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
):
    """Deterministic ODE sampler to generate x_0 from x_t"""
    if num_steps == 1:
        denoised = net(x_initial, sigma_start)
        return denoised
    last_sigma = sigma_eps
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_initial.device)

    t_steps = (
        sigma_start ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (last_sigma ** (1 / rho) - sigma_start ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )

    x_next = x_initial
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_hat = t_cur
        x_hat = x_cur
        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next
