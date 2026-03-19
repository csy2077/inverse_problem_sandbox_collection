import os
import re
import io
import hashlib
import glob
import tempfile
import urllib
import requests
import logging
import pickle
from typing import Any, Dict
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import lmdb
import sigpy as sp
from piq import psnr, SSIMLoss


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

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

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
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise

    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name if url_name else "file")
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            f.write(url_data)
        if return_filename:
            return cache_file

    return io.BytesIO(url_data)

def parse_int_list(s):
    if isinstance(s, list): return s
    if isinstance(s, int): return [s]
    if s is None: return None
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

_model_dict = {
    'DhariwalUNet': DhariwalUNet,
}

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
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
    
    def close(self):
        pass

class MultiCoilMRI(BaseOperator):
    def __init__(self, total_lines=128, acceleration_ratio=8, pattern='random', orientation='vertical', mask_fname=None, mask_seed=0, device='cuda', sigma_noise=0.0):
        '''
        MRI forward operator
        Args:
            - mask: sampling mask
            - device: device to run the operator
            - dtype: data type
        '''
        super(MultiCoilMRI, self).__init__(sigma_noise=sigma_noise)
        if mask_fname is None:
            if 1 < acceleration_ratio <= 6:
                # Keep 8% of center samples
                acs_lines = np.floor(0.08 * total_lines).astype(int)
            else:
                # Keep 4% of center samples
                acs_lines = np.floor(0.04 * total_lines).astype(int)
            mask = self.get_mask(acs_lines, total_lines, acceleration_ratio, pattern, seed=mask_seed)
        else:
            mask = np.load(mask_fname)
        if orientation == 'vertical':
            mask = mask[None, None, None, :].astype(bool)
        elif orientation == 'horizontal':
            mask = mask[None, None, :, None].astype(bool)
        else:
            raise NotImplementedError
        self.mask = torch.from_numpy(mask).to(device)
        self.device = device

    @staticmethod
    # Phase encode random mask generator
    def get_mask(acs_lines=30, total_lines=384, R=1, pattern='random', seed=0):
        np.random.seed(seed)

        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    def unnormalize(self, gen_img):
        return gen_img

    def normalize(self, gen_img):
        return gen_img

    @staticmethod
    def ifft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def fft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x

    def __call__(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.maps = data['maps']
        self.masked_kspace = self.mask * data['kspace']
        self.estimated_mvue = MultiCoilMRIData.get_mvue(
            self.masked_kspace.cpu().numpy(),
            self.maps.cpu().numpy()
        )
        return torch.view_as_real(self.masked_kspace)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image).to(torch.float64)
        coils = self.maps * torch.view_as_complex(image.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
        return torch.view_as_real(self.mask * self.fft(coils))

class MultiCoilMRIData:
    @staticmethod
    def get_rss(kspace):
        return np.sqrt(np.sum(np.square(np.abs(sp.ifft(kspace))), axis=0, keepdims=True))

    @staticmethod
    def get_mvue(kspace, s_maps):
        ''' Get mvue estimate from coil measurements '''
        return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

    @staticmethod
    def unnormalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img * scaling

    @staticmethod
    def normalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img / scaling

class MultiCoilMRILMDBData(Dataset):
    def __init__(self, root, image_size, mvue_only=False, slice_range=[5, -5], id_list=None, simulated_kspace=False):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.mvue_only = mvue_only
        self.simulated_kspace = simulated_kspace
        if not mvue_only:
            kspace_env = lmdb.open(str(self.root / "kspace"), readonly=True, lock=False, create=False)
            s_maps_env = lmdb.open(str(self.root / "s_maps"), readonly=True, lock=False, create=False)
            self.kspace_txn = kspace_env.begin(write=False)
            self.s_maps_txn = s_maps_env.begin(write=False)
        mvue_env = lmdb.open(str(self.root / "mvue"), readonly=True, lock=False, create=False)
        self.mvue_txn = mvue_env.begin(write=False)
        if id_list is None:
            self.length = self.mvue_txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f'{self.idx_map(idx)}'.encode('utf-8')
        mvue_bytes = self.mvue_txn.get(key)
        mvue = np.frombuffer(mvue_bytes, dtype=np.complex64)
        mvue = mvue.reshape(1, self.image_size[0], self.image_size[1])
        mvue_scaled = MultiCoilMRIData.normalize(mvue, mvue)
        if self.mvue_only:
            return np.concatenate([mvue_scaled.real, mvue_scaled.imag], axis=0)
        else:
            s_maps_bytes = self.s_maps_txn.get(key)
            s_maps = np.frombuffer(s_maps_bytes, dtype=np.complex64)
            maps = s_maps.reshape(-1, self.image_size[0], self.image_size[1])
            if self.simulated_kspace:
                gt_ksp_scaled = sp.fft(maps * mvue_scaled, axes=(-2, -1))
            else:
                kspace_bytes = self.kspace_txn.get(key)
                kspace = np.frombuffer(kspace_bytes, dtype=np.complex64)
                gt_ksp = kspace.reshape(-1, self.image_size[0], self.image_size[1])
                gt_ksp_scaled = MultiCoilMRIData.normalize(gt_ksp, mvue)
            return {
                'target': torch.view_as_real(torch.from_numpy(mvue_scaled).squeeze(0)).permute(2, 0, 1).contiguous(),
                'mvue': mvue_scaled,
                'maps': maps,
                'kspace': gt_ksp_scaled,
            }

class Algo(ABC):
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass

class ScoreMRI(Algo):
    def __init__(self, net, forward_op, scheduler_config, correct_steps=1, lr=0.16, lamb=1):
        super(ScoreMRI, self).__init__(net, forward_op)
        self.scheduler = Scheduler(**scheduler_config)
        self.correct_steps = correct_steps
        self.lr = lr
        self.lamb = lamb

    def score(self, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = self.net(x, sigma)
        return (d - x) / sigma**2

    def project(self, x, observation):
        gradient = self.forward_op.gradient(x, observation, return_loss=False)
        return x - self.lamb * gradient

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        pbar = tqdm.trange(self.scheduler.num_steps)
        x_next = torch.randn(observation.shape[0], self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next.requires_grad = True

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)
            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

            # Predictor step
            score = self.score(x_cur / self.scheduler.scaling_steps[i], sigma) / self.scheduler.scaling_steps[i]
            epsilon = torch.randn_like(x_cur)
            x_cur = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon

            # Projection
            x_cur = self.project(x_cur, observation)

            # Corrector steps
            for _ in range(self.correct_steps):
                sigma = self.scheduler.sigma_steps[i+1]

                score = self.score(x_cur / self.scheduler.scaling_steps[i+1], sigma) / self.scheduler.scaling_steps[i+1]
                epsilon = torch.randn_like(x_cur)
                lr = 2 * self.lr * torch.norm(epsilon)/torch.norm(score)
                x_cur = x_cur + lr * score + torch.sqrt(2*lr) * epsilon

                # Projection
                x_cur = self.project(x_cur, observation)
            difference = observation - self.forward_op.forward(x_cur)
            norm = torch.linalg.norm(difference)
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Avg. Error: {norm.abs().mean().cpu().item()}')
            x_next = x_cur
        return x_next

class DynamicRangePSNRLoss:
    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2: # complex input: convert to magnitude image
            yhats = torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        if ys.shape[1] == 2:
            ys = torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        return -torch.mean(torch.stack([psnr(yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0), data_range=y.max()) for yhat, y in zip(yhats, ys)]))

class DynamicRangeSSIMLoss:
    def __init__(self):
        self.ssim_loss = SSIMLoss()

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2: # complex input: convert to magnitude image
            yhats = torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        if ys.shape[1] == 2:
            ys = torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        return torch.mean(torch.stack([SSIMLoss(data_range=y.max())(yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0)) for yhat, y in zip(yhats, ys)]))

class MRIEvaluator:
    def __init__(self, forward_op=None):
        dr_psnr_loss = DynamicRangePSNRLoss()
        dr_ssim_loss = DynamicRangeSSIMLoss()
        self.eval_batch = 32
        self.metric_list = {
            'psnr': lambda x, y: -dr_psnr_loss(x, y),
            'ssim': lambda x, y: 1-dr_ssim_loss(x, y)
        }
        self.forward_op = forward_op
        self.metric_state = defaultdict(list)
        self.device = forward_op.device if forward_op is not None else 'cpu'

    def __call__(self, pred, target, observation=None):
        '''
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        '''
        metric_dict = {}
        for metric_name, metric_func in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if len(pred) != len(target):
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch: (i + 1) * self.eval_batch]
                    target_batch = target.repeat(pred_batch.shape[0], 1, 1, 1)
                    val = metric_func(pred_batch, target_batch).squeeze(-1).sum()
                    metric_dict[metric_name] += val
                metric_dict[metric_name] = metric_dict[metric_name] / pred.shape[0]
                self.metric_state[metric_name].append(metric_dict[metric_name])
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        if self.forward_op is not None and observation is not None:
            pred = pred.to(self.device)
            observation = observation.to(self.device)
            metric_dict['data misfit'] = torch.linalg.norm(self.forward_op.forward(pred) - observation).item()
            self.metric_state['data misfit'].append(metric_dict['data misfit'])
        return metric_dict

    def compute(self):
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

def main():
    # Configuration (derived from yaml configs + command line args)
    config = EasyDict({
        'tf32': True,
        'inference': True,
        'num_samples': 1,
        'compile': False,
        'seed': 0,
        'wandb': False,
        'exp_name': 'default',
        'problem': EasyDict({
            'name': 'multi-coil-mri-knee-lmdb',
            'prior': 'weights/MRI-knee.pt',  # Will be overridden by command line
            'model': {
                'sigma_noise': 0.0,
                'total_lines': 320,
                'acceleration_ratio': 4,
                'pattern': 'random',
                'mask_seed': 0,
            },
            'data': {
                'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/knee_val_lmdb',
                'image_size': [320, 320],
                'id_list': None,  # Will use only first sample
                'simulated_kspace': False,
            },
            'exp_dir': 'outputs/standalone_mri_score_mri'
        }),
        'algorithm': EasyDict({
            'name': 'ScoreMRI',
            'method': {
                'scheduler_config': {
                    'num_steps': 2000,
                    'sigma_max': 100,
                    'sigma_min': 0.01,
                    'schedule': 'vp',
                    'timestep': 'vp',
                    'scaling': 'vp',
                },
                'correct_steps': 1,
                'lr': 0.01,
                'lamb': 0.01,  # From command line: algorithm.method.lamb=0.01
            }
        }),
        'pretrain': EasyDict({
            'model': {
                'model_type': 'DhariwalUNet',
                'img_resolution': 320,
                'img_channels': 2,  # Complex data: real + imag
                'label_dim': 0,
                'model_channels': 128,
                'channel_mult': [1, 1, 1, 2, 2],
                'attn_resolutions': [16],
                'num_blocks': 1,
                'dropout': 0.0
            }
        })
    })

    # Use absolute path for weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config.problem.prior = os.path.join(script_dir, 'weights/MRI-knee.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    
    torch.manual_seed(config.seed)

    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    
    # Instantiate Problem Model (MultiCoilMRI forward operator)
    logger.info("Instantiating MultiCoilMRI problem model...")
    forward_op = MultiCoilMRI(
        **config.problem.model,
        device=device
    )

    # Instantiate Dataset - only use first sample
    logger.info("Instantiating Dataset (first sample only)...")
    testset = MultiCoilMRILMDBData(
        root=config.problem.data['root'],
        image_size=config.problem.data['image_size'],
        id_list='0',  # Only first sample
        simulated_kspace=config.problem.data['simulated_kspace']
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f"Loaded {len(testset)} test sample(s)...")

    # Load Pretrained Model
    logger.info("Loading pre-trained model...")
    ckpt_path = config.problem.prior
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        logger.info(f"Could not load pickle, trying torch.load from {ckpt_path}")
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
    logger.info(f"Loaded pre-trained model from {config.problem.prior}...")
    
    # Instantiate Algorithm (ScoreMRI)
    logger.info("Instantiating ScoreMRI algorithm...")
    algo = ScoreMRI(
        net=net,
        forward_op=forward_op,
        **config.algorithm.method
    )

    # Instantiate Evaluator
    logger.info("Instantiating MRI Evaluator...")
    evaluator = MRIEvaluator(forward_op=forward_op)

    # Inference Loop (only first sample)
    for i, data in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
                elif isinstance(val, np.ndarray):
                    data[key] = torch.from_numpy(val).to(device)
        
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')

        if config.inference:
            # Get the observation (apply forward operator with masking)
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
                logger.info(f"Loaded results from {save_path}.")
                observation = result_dict['observation']
            else:
                logger.warning(f"Result file {save_path} not found, skipping evaluation for this sample.")
                continue

        # Evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=observation)
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
    
    forward_op.close()
