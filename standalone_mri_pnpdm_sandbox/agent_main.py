import os
import copy
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from piq import psnr, SSIMLoss
import lmdb
import sigpy as sp


# --- Extracted Dependencies ---

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

def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    if isinstance(s, int):
        return [s]
    import re
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

_model_dict = {
    'DhariwalUNet': DhariwalUNet,
}

class EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5,
                 model_type='DhariwalUNet', **model_kwargs):
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
            [1 - scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])
        factor_steps = np.array(
            [2 * scaling_fn(time_steps[i]) ** 2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])
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
            sigma_fn = lambda t: np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
            sigma_derivative_fn = lambda t: (beta_d * t + beta_min) * np.exp(beta_d * t ** 2 / 2 + beta_min * t) / 2 / sigma_fn(t)
            sigma_inv_fn = lambda sigma: np.sqrt(beta_min ** 2 + 2 * beta_d * np.log(sigma ** 2 + 1)) / beta_d - beta_min / beta_d
        else:
            raise NotImplementedError
        return sigma_fn, sigma_derivative_fn, sigma_inv_fn

    def get_scaling_fn(self, schedule):
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            scaling_fn = lambda t: 1 / np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t))
            scaling_derivative_fn = lambda t: - (beta_d * t + beta_min) / 2 / np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t))
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
        return (d - x) / sigma ** 2

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
        super(MultiCoilMRI, self).__init__(sigma_noise=sigma_noise)
        if mask_fname is None:
            if 1 < acceleration_ratio <= 6:
                acs_lines = np.floor(0.08 * total_lines).astype(int)
            else:
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
    def get_mask(acs_lines=30, total_lines=384, R=1, pattern='random', seed=0):
        np.random.seed(seed)
        num_sampled_lines = np.floor(total_lines / R)
        center_line_idx = np.arange((total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2)
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            random_line_idx = np.random.choice(outer_line_idx, size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')
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
        self.estimated_mvue = MultiCoilMRIDataHelper.get_mvue(
            self.masked_kspace.cpu().numpy(),
            self.maps.cpu().numpy()
        )
        return torch.view_as_real(self.masked_kspace)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image).to(torch.float64)
        coils = self.maps * torch.view_as_complex(image.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
        return torch.view_as_real(self.mask * self.fft(coils))

class MultiCoilMRIDataHelper:
    @staticmethod
    def get_rss(kspace):
        return np.sqrt(np.sum(np.square(np.abs(sp.ifft(kspace))), axis=0, keepdims=True))

    @staticmethod
    def get_mvue(kspace, s_maps):
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
        mvue_scaled = MultiCoilMRIDataHelper.normalize(mvue, mvue)
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
                gt_ksp_scaled = MultiCoilMRIDataHelper.normalize(gt_ksp, mvue)
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

def get_exponential_decay_scheduler(num_steps, sigma_max, sigma_min, rho=0.9):
    sigma_steps = []
    sigma = sigma_max
    for i in range(num_steps):
        sigma_steps.append(sigma)
        sigma = max(sigma_min, sigma * rho)
    return sigma_steps

class LangevinDynamics:
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
            pbar.set_description(f'Iteration {step + 1}/{num_steps}. Avg. Error: {loss.sqrt().mean().cpu().item()}')
        return x

class DynamicRangePSNRLoss:
    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:
            yhats = torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        if ys.shape[1] == 2:
            ys = torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        return -torch.mean(torch.stack([psnr(yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0), data_range=y.max()) for yhat, y in zip(yhats, ys)]))

class DynamicRangeSSIMLoss:
    def __init__(self):
        self.ssim_loss = SSIMLoss()

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:
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
            'ssim': lambda x, y: 1 - dr_ssim_loss(x, y)
        }
        self.forward_op = forward_op
        self.metric_state = defaultdict(list)
        self.device = forward_op.device if forward_op is not None else 'cpu'

    def __call__(self, pred, target, observation=None):
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
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    print(f"Using device: {device}")

    # Configuration
    config = EasyDict({
        'num_samples': 1,
        'problem': EasyDict({
            'name': 'multi-coil-mri',
            'prior': '/fs-computility-new/UPDZ02_sunhe/shared/InverseBench/weights/MRI-knee.pt',
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
                'simulated_kspace': False,
            },
        }),
        'algorithm': EasyDict({
            'name': 'PnPDM',
            'method': {
                'annealing_scheduler_config': {
                    'num_steps': 10,
                    'sigma_max': 10,
                    'sigma_min': 0.02,
                    'rho': 0.93,
                },
                'diffusion_scheduler_config': {
                    'num_steps': 10,
                    'sigma_min': 1e-2,
                    'sigma_final': 0,
                    'schedule': 'linear',
                    'timestep': 'poly-7',
                },
                'lgvd_config': {
                    'num_steps': 4,
                    'lr': 5e-5,
                    'tau': 1,
                },
            }
        }),
        'pretrain': EasyDict({
            'model': {
                'model_type': 'DhariwalUNet',
                'img_resolution': 320,
                'img_channels': 2,
                'label_dim': 0,
                'model_channels': 128,
                'channel_mult': [1, 1, 1, 2, 2],
                'attn_resolutions': [16],
                'num_blocks': 1,
                'dropout': 0.0,
            }
        })
    })

    # Create output directory
    exp_dir = f"outputs/standalone_mri_pnpdm"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Output directory: {exp_dir}")

    # Instantiate Forward Operator
    print("Instantiating MultiCoilMRI forward operator...")
    forward_op = MultiCoilMRI(
        **config.problem.model,
        device=device
    )

    # Instantiate Dataset - only use first sample (id_list='0')
    print("Instantiating Dataset (first sample only)...")
    testset = MultiCoilMRILMDBData(
        root=config.problem.data['root'],
        image_size=config.problem.data['image_size'],
        simulated_kspace=config.problem.data['simulated_kspace'],
        id_list='0'  # Only first sample
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    print(f"Loaded {len(testset)} test sample(s)...")

    # Load Pre-trained Model
    print(f"Loading pre-trained model from {config.problem.prior}...")
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
    print("Model loaded successfully!")

    # Instantiate Algorithm
    print("Instantiating PnPDM algorithm...")
    algo = PnPDM(
        net=net,
        forward_op=forward_op,
        **config.algorithm.method
    )

    # Instantiate Evaluator
    print("Instantiating MRI Evaluator...")
    evaluator = MRIEvaluator(forward_op=forward_op)

    # Inference
    for i, data in enumerate(testloader):
        print(f"\n--- Processing sample {i} ---")

        # Move data to device
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
                elif isinstance(val, np.ndarray):
                    data[key] = torch.from_numpy(val).to(device)

        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')

        # Get observation
        print("Computing observation (forward model)...")
        observation = forward_op(data)
        target = data['target']

        # Run inference
        print("Running PnPDM inference...")
        recon = algo.inference(observation, num_samples=config.num_samples)

        if torch.cuda.is_available():
            print(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

        # Save results
        result_dict = {
            'observation': observation.cpu(),
            'recon': forward_op.unnormalize(recon).cpu(),
            'target': forward_op.unnormalize(target).cpu(),
        }
        torch.save(result_dict, save_path)
        print(f"Saved results to {save_path}")

        # Evaluate
        print("Evaluating results...")
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=observation)
        print(f"Metrics: {metric_dict}")

    # Final metrics
    print("\n--- Final Results ---")
    metric_state = evaluator.compute()
    print(f"Final metric results: {metric_state}")

    print("\nInference completed successfully!")
