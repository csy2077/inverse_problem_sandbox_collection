import numpy as np
import torch
from tqdm import tqdm


# --- Extracted Dependencies ---

class Scheduler:
    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear',
                 timestep='poly-7', scaling='none'):
        super().__init__()
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_final = sigma_final
        if self.sigma_final is None:
            self.sigma_final = self.sigma_min
        self.schedule = schedule
        self.timestep = timestep

        steps = np.linspace(0, 1, num_steps)
        sigma_fn, sigma_derivative_fn, sigma_inv_fn = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)
        scaling_fn, scaling_derivative_fn = self.get_scaling_fn(scaling)
        if self.schedule == 'vp':
            self.sigma_max = sigma_fn(1) * scaling_fn(1)
            
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])
        scaling_steps = np.array([scaling_fn(t) for t in time_steps])
        
        scaling_factor = np.array(
            [1 -  scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        
        factor_steps = np.array(
            [2 * scaling_fn(time_steps[i])**2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
             i in range(num_steps)])
        self.sigma_steps, self.time_steps, self.factor_steps, self.scaling_factor, self.scaling_steps = sigma_steps, time_steps, factor_steps, scaling_factor, scaling_steps
        self.factor_steps = [max(f, 0) for f in self.factor_steps]

    def get_sigma_fn(self, schedule):
        if schedule == 'sqrt':
            sigma_fn = lambda t: np.sqrt(t)
            sigma_derivative_fn = lambda t: 1 / 2 / np.sqrt(t)
            sigma_inv_fn = lambda sigma: sigma ** 2
        elif schedule == 'linear':
            sigma_fn = lambda t: t
            sigma_derivative_fn = lambda t: 1
            sigma_inv_fn = lambda t: t
        elif schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            sigma_fn = lambda t: np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t) - 1)
            sigma_derivative_fn = lambda t: (beta_d * t + beta_min)*np.exp(beta_d * t**2/2 + beta_min * t) / 2 / sigma_fn(t)
            sigma_inv_fn = lambda sigma: np.sqrt(beta_min**2 + 2*beta_d*np.log(sigma**2 + 1))/beta_d - beta_min/beta_d
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_scaling_fn(self, schedule):
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            scaling_fn = lambda t: 1/ np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
            scaling_derivative_fn = lambda t: - (beta_d * t + beta_min)/ 2 / np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
        else:
            scaling_fn = lambda t: 1
            scaling_derivative_fn = lambda t: 0
        return scaling_fn, scaling_derivative_fn

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        elif timestep == 'vp':
            get_time_step_fn = lambda r: 1 - r * (1 - 1e-3)
        else:
            raise NotImplementedError
        return get_time_step_fn

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

def run_inversion(observation: torch.Tensor, 
                  net: torch.nn.Module,
                  forward_op_params: dict,
                  scheduler_config: dict,
                  guidance_scale: float,
                  sde: bool,
                  num_samples: int,
                  device: torch.device) -> torch.Tensor:
    """
    Runs the DPS inversion algorithm using the forward_operator.
    Returns the reconstructed image.
    """
    scheduler = Scheduler(**scheduler_config)
    
    if num_samples > 1:
        observation = observation.repeat(num_samples, 1, 1, 1)
    
    x_initial = torch.randn(
        num_samples, 
        net.img_channels, 
        net.img_resolution, 
        net.img_resolution, 
        device=device
    ) * scheduler.sigma_max
    
    x_next = x_initial
    x_next.requires_grad = True
    
    pbar = tqdm(range(scheduler.num_steps))
    
    for i in pbar:
        x_cur = x_next.detach().requires_grad_(True)
        
        sigma = scheduler.sigma_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        
        # Denoising step
        denoised = net(x_cur / scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
        
        # Compute data fidelity gradient using forward_operator
        pred_tmp = denoised.clone().detach().requires_grad_(True)
        
        # Apply forward operator to get predicted measurements
        y_pred = forward_operator(pred_tmp, forward_op_params, unnormalize=True)
        
        # Compute loss between predicted and observed measurements
        diff = y_pred - observation
        squared_diff = diff * diff.conj()
        loss = torch.sum(squared_diff.real)
        
        # Compute gradient of loss w.r.t. denoised image
        gradient = torch.autograd.grad(loss, pred_tmp)[0]
        loss_scale = loss.detach()
        
        # Backpropagate through denoiser
        ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
        ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)
        
        # Score computation
        score = (denoised - x_cur / scheduler.scaling_steps[i]) / sigma ** 2 / scheduler.scaling_steps[i]
        
        pbar.set_description(f'Iteration {i + 1}/{scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale):.4f}')
        
        # Update step
        if sde:
            epsilon = torch.randn_like(x_cur)
            x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
        else:
            x_next = x_cur * scaling_factor + factor * score * 0.5
        
        x_next = x_next - ll_grad * guidance_scale
    
    return x_next
