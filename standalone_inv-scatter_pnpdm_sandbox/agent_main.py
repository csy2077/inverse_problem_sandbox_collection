import os
import copy
import re
import warnings
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
from piq import psnr, ssim
import lmdb
from scipy.special import hankel1
from scipy.integrate import dblquad


# --- Extracted Dependencies ---

def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, list): 
        return s
    if isinstance(s, int): 
        return [s]
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def create_logger(logging_dir, main_process=True):
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

class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
    def __delattr__(self, name: str) -> None:
        del self[name]

class Scheduler:
    """Scheduler for diffusion sigma(t) and discretization step size Delta t"""

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
            [1 - scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for
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
            scaling_fn = lambda t: 1 / np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
            scaling_derivative_fn = lambda t: - (beta_d * t + beta_min) / 2 / np.sqrt(np.exp(beta_d * t**2/2 + beta_min * t))
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

    @classmethod
    def get_partial_scheduler(cls, scheduler, new_sigma_max):
        new_scheduler = copy.deepcopy(scheduler)
        num_steps = sum([s < new_sigma_max for s in scheduler.sigma_steps]) + 1

        new_scheduler.num_steps = num_steps - 1
        new_scheduler.sigma_max = new_sigma_max
        new_scheduler.sigma_steps = scheduler.sigma_steps[-num_steps:]
        new_scheduler.time_steps = scheduler.time_steps[-num_steps:]
        new_scheduler.factor_steps = scheduler.factor_steps[-num_steps:]
        new_scheduler.scaling_factor = scheduler.scaling_factor[-num_steps:]
        new_scheduler.scaling_steps = scheduler.scaling_steps[-num_steps:]
        return new_scheduler

class DiffusionSampler:
    """Diffusion sampler for reverse SDE or PF-ODE"""

    def __init__(self, scheduler, solver='euler'):
        super().__init__()
        self.scheduler = scheduler
        self.solver = solver

    def sample(self, model, x_start, SDE=False, verbose=False):
        if self.solver == 'euler':
            return self._euler(model, x_start, SDE, verbose)
        else:
            raise NotImplementedError

    def score(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (d - x) / sigma**2
    
    def _euler(self, model, x_start, SDE=False, verbose=False):
        pbar = tqdm(range(self.scheduler.num_steps)) if verbose else range(self.scheduler.num_steps)

        x = x_start
        for step in pbar:
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[step], self.scheduler.factor_steps[step], self.scheduler.scaling_factor[step]
            score = self.score(model, x / self.scheduler.scaling_steps[step], sigma) / self.scheduler.scaling_steps[step]
            if SDE:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5 
        return x

    def get_start(self, ref):
        x_start = torch.randn_like(ref) * self.scheduler.sigma_max
        return x_start

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
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = torch.autograd.grad(loss, pred_tmp)[0]
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad

    def loss(self, pred, observation, **kwargs):
        return (self.forward(pred) - observation).square().flatten(start_dim=1).sum(dim=1)
    
    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale
    
    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift

def full_propagate_to_sensor(f, utot_dom_set, sensor_greens_function_set, dx, dy):
    """
    Propagate all the total fields to the sensors.
    """
    num_trans = utot_dom_set.shape[2]
    num_rec = sensor_greens_function_set.shape[2]
    contSrc = f[0, 0].unsqueeze(-1) * utot_dom_set
    conjSrc = torch.conj(contSrc).reshape(-1, num_trans)
    sensor_greens_func = sensor_greens_function_set.reshape(-1, num_rec)
    uscat_pred_set = dx * dy * torch.matmul(conjSrc.T, sensor_greens_func)
    return uscat_pred_set

def generate_em_functions(p):
    # Meshgrid the pixel locations
    XPix, YPix = np.meshgrid(p['x'], p['y'])
    
    # Hankel function
    hank_fun = lambda x: 1j * 0.25 * hankel1(0, x)
    
    # Locations of transmitters and receivers
    transmitter_angles = np.linspace(0, 359, p['numTrans']) * np.pi / 180
    x_transmit = p['sensorRadius'] * np.cos(transmitter_angles)
    y_transmit = p['sensorRadius'] * np.sin(transmitter_angles)

    receiver_angles = np.linspace(0, 359, p['numRec']) * np.pi / 180
    x_receive = p['sensorRadius'] * np.cos(receiver_angles)
    y_receive = p['sensorRadius'] * np.sin(receiver_angles)
    
    p['receiverMask'] = np.ones((p['numTrans'], p['numRec']))
    
    diff_x_rp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(x_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_rp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(y_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_rec_to_pix = np.sqrt(diff_x_rp**2 + diff_y_rp**2)

    diff_x_tp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(x_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_tp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(y_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_trans_to_pix = np.sqrt(diff_x_tp**2 + diff_y_tp**2)
    
    # Input fields
    p['uincDom'] = hank_fun(p['kb'] * distance_trans_to_pix)
    
    # Sensor Green's functions
    sensor_greens_function = hank_fun(p['kb'] * distance_rec_to_pix)
    p['sensorGreensFunction'] = (p['kb']**2) * sensor_greens_function
    
    # Domain Green's functions
    x_green = np.arange(-p['Nx'], p['Nx']) * p['dx']                        
    y_green = np.arange(-p['Ny'], p['Ny']) * p['dy']
    
    XGreen, YGreen = np.meshgrid(x_green, y_green)
    R = np.sqrt(XGreen**2 + YGreen**2)
    
    domain_greens_function = hank_fun(p['kb'] * R)
    
    def integrand_real(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).real)
        
    def integrand_imag(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).imag)
    
    Ny = p['Ny']
    Nx = p['Nx']
    dx = p['dx']
    dy = p['dy']
    domain_greens_function[Ny, Nx] = dblquad(
            integrand_real,
            -dx/2, dx/2, -dy/2, dy/2
        )[0] / (dx * dy)
    domain_greens_function[Ny, Nx] += (dblquad(
        integrand_imag,
        -dx/2, dx/2, -dy/2, dy/2
    )[0] / (dx * dy)) * 1j
    
    p['domainGreensFunction'] = (p['kb']**2) * domain_greens_function
    
    return p

def construct_parameters(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=360, numTrans=60, sensorRadius=1.6,
                         device='cuda'):
    em = {}

    em['Lx'] = Lx
    em['Ly'] = Ly
    em['Nx'] = Nx
    em['Ny'] = Ny
    em['dx'] = em['Lx'] / em['Nx']
    em['dy'] = em['Ly'] / em['Ny']
    em['x'] = np.linspace(-em['Nx']/2, em['Nx']/2 - 1, em['Nx']) * em['dx']
    em['y'] = np.linspace(-em['Ny']/2, em['Ny']/2 - 1, em['Ny']) * em['dy']
    em['c'] = 299792458
    em['lambda'] = em['dx'] * wave
    em['freq'] = em['c'] / em['lambda'] / 1e9
    em['numRec'] = numRec
    em['numTrans'] = numTrans
    em['sensorRadius'] = sensorRadius
    em['kb'] = 2 * np.pi / em['lambda']

    em = generate_em_functions(em)
    return (torch.from_numpy(em['domainGreensFunction']).to(device).unsqueeze(-1), 
            torch.from_numpy(em['sensorGreensFunction']).to(device).unsqueeze(-1), 
            torch.from_numpy(em['uincDom']).to(device).unsqueeze(-1), 
            torch.from_numpy(em['receiverMask']).unsqueeze(-1))

class InverseScatter(BaseOperator):
    
    def __init__(self, Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, 
                 numRec=360, numTrans=60, sensorRadius=1.6, svd=True, **kwargs):
        super(InverseScatter, self).__init__(**kwargs)
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.numRec = numRec
        self.numTrans = numTrans
        self.domain_greens_function_set, self.sensor_greens_function_set, self.uinc_dom_set, self.receiver_mask_set = \
        construct_parameters(Lx, Ly, Nx, Ny, wave, numRec, numTrans, sensorRadius, self.device)
        
        self.sensor_greens_function_set = self.sensor_greens_function_set.to(torch.complex128)
        self.uinc_dom_set = self.uinc_dom_set.to(torch.complex128)

        if svd:
            self.compute_svd()

        
    def forward(self, f, unnormalize=True):
        '''
        Parameters:
            f - permittivity, (batch_size, 1, Ny, Nx), torch.Tensor, float32
            
        Returns:
            uscat_pred_set - (batch_size, numTrans, numRec) predicted scattered fields, torch.Tensor, complex64
        '''
        f = f.to(torch.float64)
        if unnormalize:
            f = self.unnormalize(f)
        # Linear inverse scattering
        uscat_pred_set = full_propagate_to_sensor(f, self.uinc_dom_set[..., 0], self.sensor_greens_function_set[..., 0], 
                                                  self.dx, self.dy)
        return uscat_pred_set.unsqueeze(0)
    
    def loss(self, pred, observation):
        '''
        Parameters:
            pred - predicted permittivity, (batch_size, 1, Ny, Nx), torch.Tensor, float32
            observation - actual observation, (1, numTrans, numRec), torch.Tensor, complex64
        Returns:
            loss - data consistency loss, (batch_size,), scalar, float32
        '''
        uscat_pred_set = self.forward(pred)
        diff = uscat_pred_set - observation
        squared_diff = diff * diff.conj()
        loss = torch.sum(squared_diff, dim=(1, 2)).to(torch.float64)
        return loss
        
    def compute_svd(self):
        '''
        Compute SVD of the forward operator A.
        '''
        path = 'cache/inv-scatter_numT_{}_numR_{}'.format(self.numTrans, self.numRec)
        if os.path.exists(path + '/matrix.pt'):
            print('Loading SVD from cache.')
            self.U = torch.load(os.path.join(path, 'U.pt'))
            self.Sigma = torch.load(os.path.join(path, 'S.pt'))
            self.V_t = torch.load(os.path.join(path, 'Vt.pt'))
            self.A = torch.load(os.path.join(path, 'matrix.pt'))
            if os.path.exists(path + '/matrix_inv.pt'):
                self.A_inv = torch.load(os.path.join(path, 'matrix_inv.pt'))
            else:
                self.A_inv = torch.linalg.pinv(self.A)
                torch.save(self.A_inv, os.path.join(path, 'matrix_inv.pt'))
        else:
            print('Computing SVD... This may take 10-20 minutes for the first time.')
            T = self.uinc_dom_set[..., 0].flatten(0,1)
            R = self.sensor_greens_function_set[..., 0].reshape(-1, self.numRec)
            A = torch.cat([R.T@torch.conj(torch.diag(T[:,i])) for i in range(T.shape[-1])], dim=0) * self.dx * self.dy
            A = torch.view_as_real(A).permute(0,2,1).flatten(0,1)
            U, Sigma, V = torch.svd(A)
            self.U = U
            self.Sigma = Sigma
            self.V_t = V.T
            self.A = A
            self.A_inv = torch.linalg.pinv(A)
            os.makedirs(path, exist_ok=True)
            torch.save(self.U, os.path.join(path, 'U.pt'))
            torch.save(self.Sigma, os.path.join(path, 'S.pt'))
            torch.save(self.V_t, os.path.join(path, 'Vt.pt'))
            torch.save(self.A, os.path.join(path, 'matrix.pt'))
            torch.save(self.A_inv, os.path.join(path, 'matrix_inv.pt'))

    def Vt(self, x):
        x = x.to(torch.float64)
        return (self.V_t @ x.flatten(-3)[...,None]).squeeze(-1)
    
    def V(self, x):
        return (self.V_t.T @ x[...,None].to(torch.float64)).reshape(-1, 1, self.Ny, self.Nx).to(torch.float32)
    
    def Ut(self, x):
        x = torch.view_as_real(x)
        return (self.U.T @ x.flatten(-3)[...,None]).squeeze(-1)
    
    def pseudo_inverse(self, x):
        return self.normalize((self.A_inv @ torch.view_as_real(x).flatten()).reshape(-1, self.Ny, self.Nx))

    @property
    def M(self):
        return (self.Sigma.abs() > 1e-3).float()
    
    @property
    def S(self):
        return self.Sigma

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
        return (data - self.mean) / (2 * self.std)

    def unnormalize(self, data):
        return data * 2 * self.std + self.mean

class Algo(ABC):
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass

def get_exponential_decay_scheduler(num_steps, sigma_max, sigma_min, rho=0.9):
    sigma_steps = []
    sigma = sigma_max
    for i in range(num_steps):
        sigma_steps.append(sigma)
        sigma = max(sigma_min, sigma * rho)
    return sigma_steps

class LangevinDynamics:
    """Langevin Dynamics sampling method."""

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=1):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = 1.0
        if self.lr_min_ratio != lr_min_ratio:
            warnings.warn('lr_min_ratio is not used in the current implementation.')

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False):
        pbar = tqdm(range(self.num_steps)) if verbose else range(self.num_steps)
        lr = self.get_lr(ratio)
        x0hat = x0hat.detach()
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        for _ in pbar:
            optimizer.zero_grad()

            gradient = operator.gradient(x, measurement) / (2 * self.tau ** 2)
            gradient += (x - x0hat) / sigma ** 2
            x.grad = gradient

            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x)
                x.data = x.data + np.sqrt(2 * lr) * epsilon

            if torch.isnan(x).any():
                return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio):
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

class PnPDM(Algo):
    def __init__(self, net, forward_op, annealing_scheduler_config={}, diffusion_scheduler_config={}, lgvd_config={}):
        super(PnPDM, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op

        self.annealing_sigmas = get_exponential_decay_scheduler(**annealing_scheduler_config)
        self.base_diffusion_scheduler = Scheduler(**diffusion_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(**lgvd_config)

    def inference(self, observation, num_samples=1, verbose=True):
        device = self.forward_op.device
        num_steps = len(self.annealing_sigmas)
        pbar = tqdm(range(num_steps)) if verbose else range(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)
        for step in pbar:
            sigma = self.annealing_sigmas[step]
            # 1. langevin dynamics
            z = self.lgvd.sample(x, self.forward_op, observation, sigma, step / num_steps)

            # 2. reverse diffusion
            diffusion_scheduler = Scheduler.get_partial_scheduler(self.base_diffusion_scheduler, sigma)
            sampler = DiffusionSampler(diffusion_scheduler)
            x = sampler.sample(self.net, z, SDE=True, verbose=False)

            loss = self.forward_op.loss(x, observation)
            if verbose:
                pbar.set_description(f'Iteration {step + 1}/{num_steps}. Avg. Error: {loss.sqrt().mean().cpu().item():.4f}')
        return x

class InverseScatterEvaluator:
    def __init__(self, forward_op=None):
        self.eval_batch = 32
        self.metric_list = {
            'psnr': lambda x, y: psnr(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none'),
            'ssim': lambda x, y: ssim(x.clip(0, 1), y.clip(0, 1), data_range=1.0, reduction='none')
        }
        self.forward_op = forward_op
        self.device = forward_op.device if forward_op is not None else 'cpu'
        self.metric_state = {key: [] for key in self.metric_list.keys()}

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            if pred.shape != target.shape:
                val = metric_func(pred, target.repeat(pred.shape[0], 1, 1, 1)).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        return metric_dict

    def compute(self):
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

def main():
    """
    Configuration from command line:
      problem=inv-scatter
      algorithm=pnpdm
      pretrain=inv-scatter
    """
    
    # Config based on inv-scatter.yaml, pnpdm.yaml (blackhole section), and inv-scatter pretrain
    config = EasyDict({
        'tf32': True,
        'inference': True,
        'num_samples': 1,
        'compile': False,
        'seed': 0,
        'exp_name': 'standalone_inv-scatter_pnpdm',
        
        # Problem config (from inv-scatter.yaml)
        'problem': EasyDict({
            'name': 'inverse-scatter-linear',
            'prior': 'weights/inv-scatter-5m.pt',
            'model': {
                'Lx': 0.18,
                'Ly': 0.18,
                'Nx': 128,
                'Ny': 128,
                'wave': 6,
                'numRec': 360,
                'numTrans': 20,
                'sensorRadius': 1.6,
                'sigma_noise': 0.0001,
                'unnorm_shift': 1.0,
                'unnorm_scale': 0.5
            },
            'data': {
                'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/inv-scatter-test',
                'resolution': 128,
                'mean': 0.5,
                'std': 0.25,
                'id_list': '0'  # Only first test sample
            },
            'exp_dir': 'exps/inference/inv-scatter-linear'
        }),
        
        # Algorithm config (from pnpdm.yaml - blackhole section as default)
        'algorithm': EasyDict({
            'name': 'PnPDM',
            'method': {
                # Annealing scheduler
                'annealing_scheduler_config': {
                    'num_steps': 10,
                    'sigma_max': 10,
                    'sigma_min': 0.02,
                    'rho': 0.93
                },
                # Diffusion scheduler
                'diffusion_scheduler_config': {
                    'num_steps': 10,
                    'sigma_min': 1e-2,
                    'sigma_final': 0,
                    'schedule': 'linear',
                    'timestep': 'poly-7'
                },
                # LGVD config
                'lgvd_config': {
                    'num_steps': 4,
                    'lr': 5e-5,
                    'tau': 1
                }
            }
        }),
        
        # Pretrain config (from inv-scatter.yaml pretrain section)
        'pretrain': EasyDict({
            'model': {
                'model_type': 'DhariwalUNet',
                'img_resolution': 128,
                'img_channels': 1,
                'label_dim': 0,
                'model_channels': 128,
                'channel_mult': [1, 1, 1, 2, 2],
                'attn_resolutions': [16],
                'num_blocks': 1,
                'dropout': 0.0
            }
        })
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    
    torch.manual_seed(config.seed)

    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    
    # Instantiate Problem Model (InverseScatter)
    logger.info("Instantiating InverseScatter problem model...")
    forward_op = InverseScatter(
        **config.problem.model,
        device=device
    )

    # Instantiate Dataset (only first sample)
    logger.info("Instantiating Dataset (first test sample only)...")
    testset = LMDBData(
        root=config.problem.data['root'],
        resolution=config.problem.data['resolution'],
        mean=config.problem.data['mean'],
        std=config.problem.data['std'],
        id_list=config.problem.data['id_list']
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f"Loaded {len(testset)} test sample(s)...")

    # Load Pretrained Model
    logger.info("Loading pre-trained model...")
    ckpt_path = config.problem.prior
    
    net = EDMPrecond(**config.pretrain.model)
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
    
    logger.info(f"Loaded pre-trained model from {ckpt_path}...")
    
    # Instantiate Algorithm (PnPDM)
    logger.info("Instantiating PnPDM algorithm...")
    algo = PnPDM(
        net=net,
        forward_op=forward_op,
        **config.algorithm.method
    )

    # Instantiate Evaluator
    logger.info("Instantiating InverseScatterEvaluator...")
    evaluator = InverseScatterEvaluator(forward_op=forward_op)

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
            # get the observation
            observation = forward_op(data)
            target = data['target']
            
            logger.info(f'Running inference on test sample {data_id}...')
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
                observation = result_dict['observation'].to(device)
                logger.info(f"Loaded results from {save_path}.")
            else:
                logger.warning(f"Result file {save_path} not found, skipping evaluation for this sample.")
                continue

        # evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'].to(device), target=result_dict['target'].to(device), observation=observation)
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
