import os
import re
import io
import math
import glob
import uuid
import json
import pickle
import hashlib
import logging
import tempfile
import urllib
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torch.fft as fft
from tqdm import tqdm
import lmdb


# --- Extracted Dependencies ---

_cache_dir = 'cache'

def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        import requests
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    import requests
    import html
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

def parse_int_list(s):
    """Parse a comma separated list of numbers or ranges and return a list of ints."""
    if isinstance(s, list): return s
    if isinstance(s, int): return [s]
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    def __delattr__(self, name: str) -> None:
        del self[name]

def create_logger(logging_dir, main_process=True):
    """Create a logger that writes to a log file and stdout."""
    if not main_process:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

_model_dict = {
    'DhariwalUNet': DhariwalUNet,
}

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        sigma_min       = 0,
        sigma_max       = float('inf'),
        sigma_data      = 0.5,
        model_type      = 'DhariwalUNet',
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = _model_dict[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class BaseOperator(ABC):
    def __init__(self, sigma_noise=0.0, unnorm_shift=0.0, unnorm_scale=1.0, device='cuda'):
        self.sigma_noise = sigma_noise
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

    @abstractmethod
    def forward(self, inputs, **kwargs):
        pass

    def __call__(self, inputs: Dict, **kwargs):
        target = inputs['target']
        out = self.forward(target, **kwargs)
        return out + self.sigma_noise * torch.randn_like(out)

    def gradient(self, pred, observation, return_loss=False):
        from torch.autograd import grad
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = grad(loss, pred_tmp)[0]
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad

    def loss(self, pred, observation, **kwargs):
        return (self.forward(pred) - observation).square().flatten(start_dim=1).sum(dim=1)
    
    def loss_m(self, measurements, observation):
        return (measurements - observation).square().flatten(start_dim=1).sum(dim=1)
    
    @torch.enable_grad()
    def gradient_m(self, measurements, observation):
        from torch.autograd import grad
        mea_tmp = measurements.clone().detach().requires_grad_(True)
        loss = self.loss_m(mea_tmp, observation).sum()
        grad_m = grad(loss, mea_tmp)[0]
        return grad_m   
        
    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale
    
    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift
    
    def close(self):
        pass

class NavierStokes2d(object):
    def __init__(self, s1, s2, 
                 L1=2*math.pi, L2=2*math.pi,
                 Re=100.0, 
                 device=None, dtype=torch.float64):
        """
        Pseudo-spectral solver for 2D Navier-Stokes equation
        Args:
            - s1, s2: spatial resolution
            - L1, L2: spatial domain
            - Re: Reynolds number
            - device: device to run the solver
            - dtype: data type
        """
        self.s1 = s1
        self.s2 = s2
        self.L1 = L1
        self.L2 = L2
        self.Re = Re
        self.h = 1.0/max(s1, s2)

        # Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.zeros((1,)),
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        # Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        # Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        # Dealiasing mask using 2/3 rule
        self.dealias = (k1**2 + k2**2 <= (s1/3)**2 + (s2/3)**2).type(dtype).to(device)
        # Ensure mean zero
        self.dealias[0,0] = 0.0

    def stream_function(self, w_h, real_space=False):
        """Compute stream function from vorticity (Fourier space)"""
        psi_h = self.inv_lap*w_h
        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    def velocity_field(self, stream_f, real_space=True):
        """Compute velocity field from stream function (Fourier space)"""
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f
        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    def nonlinear_term(self, w_h, f_h=None):
        """Compute non-linear term + forcing from given vorticity (Fourier space)"""
        dealias_w_h = w_h*self.dealias
        w = fft.irfft2(dealias_w_h, s=(self.s1, self.s2))
        q, v = self.velocity_field(self.stream_function(dealias_w_h, real_space=False), real_space=True)
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))
        if f_h is not None:
            nonlin += f_h
        return nonlin
    
    def time_step(self, q, v, f, Re):
        """Compute adaptive time step"""
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))
        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def solve(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        """Solve the Navier-Stokes equation"""
        GG = (1.0/Re)*self.G
        w_h = fft.rfft2(w)
        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re)

        time = 0.0
        while time < T:
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            nonlin1 = self.nonlinear_term(w_h, f_h)
            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)
            time += current_delta_t

            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.solve(w, f, T, Re, adaptive, delta_t)

class ForwardNavierStokes2d(BaseOperator):
    """Forward operator for 2D Navier-Stokes equation"""
    def __init__(self, 
                 resolution=128, L=2 * math.pi,
                 forward_time=1.0,
                 Re=200.0, 
                 downsample_factor=2,
                 dtype=torch.float32,
                 delta_t=1e-2,
                 adaptive=True, 
                 **kwargs):
        super(ForwardNavierStokes2d, self).__init__(**kwargs)
        self.dtype = dtype
        self.solver = NavierStokes2d(resolution, resolution, L, L, device=self.device, dtype=dtype)
        self.force = self.get_forcing(resolution, L)
        self.downsample_factor = downsample_factor
        self.forward_time = forward_time
        self.Re = Re
        self.delta_t = delta_t
        self.adaptive = adaptive

    def get_forcing(self, resolution, L):
        t = torch.linspace(0, L, resolution+1, device=self.device, dtype=self.dtype)[0:-1]
        _, y = torch.meshgrid(t, t, indexing='ij')
        return - 4 * torch.cos(4.0 * y)

    @torch.no_grad()
    def __call__(self, data: Dict[str, torch.Tensor], unnormalize=True):
        x = data['target']
        sol = self.forward(x, unnormalize)
        sol += self.sigma_noise * torch.randn_like(sol)
        return sol
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, unnormalize=True):
        if unnormalize:
            raw_u = self.unnormalize(x)
        else:
            raw_u = x
        sol = self.solver.solve(raw_u.squeeze(1), self.force, self.forward_time, self.Re, adaptive=self.adaptive, delta_t=self.delta_t)
        sol = sol[..., ::self.downsample_factor, ::self.downsample_factor]
        return sol.unsqueeze(1).to(torch.float32)

class Algo(ABC):
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass

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

class EnKG(Algo):
    """
    Ensemble Kalman Diffusion Guidance
    Paper: Ensemble kalman diffusion guidance: A derivative-free method for inverse problems
    Official implementation: https://github.com/devzhk/enkg-pytorch
    """
    def __init__(self, 
                 net, 
                 forward_op,
                 guidance_scale, 
                 num_steps, 
                 num_updates, 
                 sigma_max,
                 sigma_min,
                 num_samples=1024,
                 threshold=0.1,
                 batch_size=128,
                 lr_min_ratio=0.0,
                 rho: int=7, 
                 factor: int=4):
        super(EnKG, self).__init__(net, forward_op)
        self.rho = rho
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.threshold = threshold
        self.num_samples = num_samples
        self.lr_min_ratio = lr_min_ratio
        self.factor = factor

    @torch.no_grad()
    def inference(self, observation, num_samples=1):
        device = self.forward_op.device
        x_initial = torch.randn(self.num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.sigma_max
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=device)

        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )

        num_batches = x_initial.shape[0] // self.batch_size
        x_next = x_initial
        denoised = torch.zeros_like(x_initial)

        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:]))
        ):
            x_cur = x_next

            # Update the ensemble particles
            if i < (self.num_steps - int(0.5 * self.threshold)) and i > self.threshold:
                x_hat, t_hat = self.update_particles(
                    x_cur,
                    observation,
                    num_steps=min(1 + (self.num_steps - i) // self.factor, 20),
                    sigma_start=t_cur,
                    guidance_scale=self.get_lr(i),
                )
            else:
                t_hat = t_cur
                x_hat = x_cur

            # Batched network forward
            for j in range(num_batches):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                denoised[start:end] = self.net(x_hat[start:end], t_hat)

            # Euler step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
        return x_next
    
    def get_lr(self, i):
        if self.lr_min_ratio > 0.0:
            return self.guidance_scale * (1 - self.lr_min_ratio) * (self.num_steps - i) / self.num_steps + self.lr_min_ratio
        else:
            return self.guidance_scale
        
    @torch.no_grad()
    def update_particles(self, particles, observation, num_steps, sigma_start, guidance_scale=1.0):
        x0s = torch.zeros_like(particles)
        num_batchs = particles.shape[0] // self.batch_size
        N, *spatial = particles.shape
        t_hat = sigma_start

        for j in range(self.num_updates):
            # Get x0 for each particle
            for i in range(num_batchs):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                x0s[start:end] = ode_sampler(
                    self.net,
                    particles[start:end],
                    num_steps=num_steps,
                    sigma_start=sigma_start,
                )
            # Get measurement for each particle
            ys = self.forward_op.forward(x0s)

            # Difference from the mean
            xs_diff = particles - particles.mean(dim=0, keepdim=True)
            ys_diff = ys - ys.mean(dim=0, keepdim=True)
            ys_err = 0.5 * self.forward_op.gradient_m(ys, observation)

            coef = (
                torch.matmul(
                    ys_err.reshape(ys_err.shape[0], -1),
                    ys_diff.reshape(ys_diff.shape[0], -1).T,
                )
                / particles.shape[0]
            )
            dxs = coef @ xs_diff.reshape(N, -1)
            lr = guidance_scale / torch.linalg.matrix_norm(coef)
            particles = particles - lr * dxs.reshape(N, *spatial)

        return particles, t_hat

class Evaluator(ABC):
    def __init__(self, metric_list, forward_op=None, data_misfit=False):
        self.metric_list = metric_list
        self.forward_op = forward_op
        self.data_misfit = data_misfit
        if data_misfit:
            assert forward_op is not None, "forward_op must be provided for data misfit evaluation"
        self.device = forward_op.device if forward_op is not None else 'cpu'
        self.metric_state = {key: [] for key in metric_list.keys()}
        if data_misfit:
            self.metric_state['data misfit'] = []

    def eval_data_misfit(self, pred, observation):
        data_misfit = self.forward_op.loss(pred, observation, unnormalize=False)
        return torch.sqrt(data_misfit)

    @abstractmethod
    def __call__(self, pred, target, observation=None, forward_op=None):
        pass

    def compute(self):
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

def relative_l2(pred, target):
    """Compute relative L2 error"""
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2

class NavierStokes2dEvaluator(Evaluator):
    def __init__(self, forward_op):
        metric_list = {'relative l2': relative_l2}
        super(NavierStokes2dEvaluator, self).__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        return metric_dict

class LMDBData(Dataset):
    def __init__(self, root, 
                 resolution=128,
                 raw_resolution=128,
                 num_channels=1,
                 norm=True,
                 mean=0.0, std=5.0, id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.raw_resolution = raw_resolution
        self.num_channels = num_channels
        self.norm = norm
        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.idx_map(idx)
        key = f'{idx}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(self.num_channels, self.raw_resolution, self.raw_resolution)
        if self.resolution != self.raw_resolution:
            img = TF.resize(torch.from_numpy(img.copy()), self.resolution, antialias=True)
        if self.norm:
            img = self.normalize(img)
        return {'target': img}

    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def normalize(self, data):
        # By default, normalize to zero mean and 0.5 std
        return (data - self.mean) / (2 * self.std)

    def unnormalize(self, data):
        return data * 2 * self.std + self.mean

def load_config(json_path):
    """Load configuration from JSON file"""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert to EasyDict recursively
    def to_easydict(d):
        if isinstance(d, dict):
            return EasyDict({k: to_easydict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [to_easydict(v) for v in d]
        else:
            return d
    
    return to_easydict(config_dict)

def resolve_path(script_dir, path):
    """Resolve a path relative to the script directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)

def main():
    # Determine script directory - ALL paths resolved relative to this
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set cache directory relative to script
    global _cache_dir
    _cache_dir = os.path.join(script_dir, 'cache')
    
    # Load configuration from JSON (relative path in data_standalone folder)
    config_path = os.path.join(script_dir, 'data_standalone', 'standalone_navier_stokes_enkg.json')
    config = load_config(config_path)
    
    # Handle sigma_max as string "inf"
    if isinstance(config.pretrain.model.get('sigma_max'), str) and config.pretrain.model.sigma_max == 'inf':
        config.pretrain.model.sigma_max = float('inf')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    
    torch.manual_seed(config.seed)

    # Resolve exp_dir relative to script directory
    exp_dir = resolve_path(script_dir, os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name))
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    
    # Instantiate Problem Model
    logger.info("Instantiating NavierStokes problem model...")
    forward_op = ForwardNavierStokes2d(
        resolution=config.problem.model['resolution'],
        forward_time=config.problem.model['forward_time'],
        Re=config.problem.model['Re'],
        downsample_factor=config.problem.model['downsample_factor'],
        sigma_noise=config.problem.model['sigma_noise'],
        unnorm_scale=config.problem.model['unnorm_scale'],
        unnorm_shift=config.problem.model['unnorm_shift'],
        adaptive=config.problem.model['adaptive'],
        delta_t=config.problem.model['delta_t'],
        device=device
    )

    # Resolve data root relative to script directory
    data_root = resolve_path(script_dir, config.problem.data['root'])
    
    # Instantiate Dataset
    logger.info("Instantiating Dataset...")
    testset = LMDBData(
        root=data_root,
        resolution=config.problem.data['resolution'],
        raw_resolution=config.problem.data.get('raw_resolution', config.problem.data['resolution']),
        num_channels=config.problem.data.get('num_channels', 1),
        mean=config.problem.data['mean'],
        std=config.problem.data['std'],
        id_list=config.problem.data['id_list']
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f"Loaded {len(testset)} test samples...")

    # Load Pretrained Model - resolve relative to script directory
    logger.info("Loading pre-trained model...")
    ckpt_path = resolve_path(script_dir, config.problem.prior)
    
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        logger.info(f"Could not load pickle, trying torch.load from {ckpt_path}")
        net = EDMPrecond(
            img_resolution=config.pretrain.model['img_resolution'],
            img_channels=config.pretrain.model['img_channels'],
            label_dim=config.pretrain.model['label_dim'],
            use_fp16=config.pretrain.model.get('use_fp16', False),
            sigma_min=config.pretrain.model.get('sigma_min', 0),
            sigma_max=config.pretrain.model.get('sigma_max', float('inf')),
            sigma_data=config.pretrain.model.get('sigma_data', 0.5),
            model_type=config.pretrain.model['model_type'],
            model_channels=config.pretrain.model['model_channels'],
            channel_mult=config.pretrain.model['channel_mult'],
            attn_resolutions=config.pretrain.model['attn_resolutions'],
            num_blocks=config.pretrain.model['num_blocks'],
            dropout=config.pretrain.model['dropout']
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        elif 'net' in ckpt.keys():
            net.load_state_dict(ckpt['net'])
        else:
            net.load_state_dict(ckpt)
             
        net = net.to(device)

    del ckpt
    net.eval()
    if config.compile:
        net = torch.compile(net)
    logger.info(f"Loaded pre-trained model from {config.problem.prior}...")
    
    # Instantiate Algorithm
    logger.info("Instantiating EnKG algorithm...")
    algo = EnKG(
        net=net,
        forward_op=forward_op,
        guidance_scale=config.algorithm.method['guidance_scale'],
        num_updates=config.algorithm.method['num_updates'],
        num_steps=config.algorithm.method['num_steps'],
        num_samples=config.algorithm.method['num_samples'],
        sigma_max=config.algorithm.method['sigma_max'],
        sigma_min=config.algorithm.method['sigma_min'],
        threshold=config.algorithm.method['threshold'],
        batch_size=config.algorithm.method['batch_size'],
        rho=config.algorithm.method['rho'],
        factor=config.algorithm.method['factor'],
        lr_min_ratio=config.algorithm.method.get('lr_min_ratio', 0.0)
    )

    # Instantiate Evaluator
    logger.info("Instantiating Evaluator...")
    evaluator = NavierStokes2dEvaluator(forward_op=forward_op)

    # Inference Loop
    for i, data in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')

        if config.inference:
            # Get the observation
            observation = forward_op(data)
            target = data['target']
            
            logger.info(f'Running inference on test sample {data_id}...')
            # Ensure no gradient tracking during inference (critical for memory efficiency)
            with torch.no_grad():
                recon = algo.inference(observation, num_samples=config.num_samples)
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

            result_dict = {
                'observation': observation.cpu(),
                'recon': forward_op.unnormalize(recon).cpu(),
                'target': forward_op.unnormalize(target).cpu(),
            }
            torch.save(result_dict, save_path)
            logger.info(f"Saved results to {save_path}.")
        else:
            if os.path.exists(save_path):
                result_dict = torch.load(save_path)
                logger.info(f"Loaded results from {save_path}.")
            else:
                logger.warning(f"Result file {save_path} not found, skipping evaluation for this sample.")
                continue

        # Evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=result_dict['observation'])
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
