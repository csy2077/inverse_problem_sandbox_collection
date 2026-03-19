# ================================================================================
# STANDALONE SCRIPT FOR NAVIER-STOKES SCG INFERENCE
# ================================================================================
#
# Original command:
#   source .venv/bin/activate && python3 main.py problem=navier-stokes algorithm=scg \
#       pretrain=navier-stokes algorithm.method.diffusion_scheduler_config.num_steps=10 \
#       algorithm.method.num_candidates=64
#
# Original script path: main.py (relative to this script's directory)
#
# Dependencies (all paths relative to this script's directory):
#   Inputs:
#     - Config JSON: data_standalone/standalone_navier_stokes_scg.json (hard-coded parameters)
#     - LMDB Data: data_standalone/navier-stokes-test/Re200.0-t5.0 (relative path)
#     - Pretrained weights: weights/ns-5m.pt (relative path)
#   Outputs:
#     - Result file: exps/inference/navier-stokes-ds2/SCG/default/result_<id>.pt
#     - Config file: exps/inference/navier-stokes-ds2/SCG/default/config.yaml
#     - Log file: exps/inference/navier-stokes-ds2/SCG/default/log.txt
#
# Iteration/Speed-up changes:
#   - num_steps changed from 1000 to 10 (as specified in command)
#   - num_candidates changed from 512 to 64 (as specified in command)
#   - Running on single sample only (sample index 0)
#
# All high-level package calls have been inlined:
#   - algo.scg.SCG
#   - inverse_problems.navier_stokes.ForwardNavierStokes2d
#   - inverse_problems.navier_stokes.NavierStokes2d
#   - inverse_problems.base.BaseOperator
#   - models.precond.EDMPrecond
#   - models.unets.DhariwalUNet
#   - models.modules (Linear, Conv2d, GroupNorm, UNetBlock, PositionalEmbedding)
#   - training.dataset.LMDBData
#   - utils.scheduler.Scheduler
#   - eval.NavierStokes2d
#   - utils.helper (create_logger)
#
# NOTE: All paths are relative to where this script is located.
#       Run this script from its directory: python standalone_navier_stokes_scg.py
#
# ================================================================================

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Change working directory to script location for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import json
import math
import copy
import logging
import lmdb
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

# ================================================================================
# CONFIGURATION - Load from JSON (relative path in data_standalone folder)
# ================================================================================

CONFIG_JSON_PATH = "data_standalone/standalone_navier_stokes_scg.json"

with open(CONFIG_JSON_PATH, 'r') as f:
    CONFIG = json.load(f)

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def parse_int_list(s):
    """Parse a comma separated list of numbers or ranges and return a list of ints."""
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
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def create_logger(logging_dir, main_process=True):
    """Create a logger that writes to a log file and stdout."""
    if not main_process:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    else:
        logger = logging.getLogger(__name__ + str(id(logging_dir)))
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(f"{logging_dir}/log.txt")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# ================================================================================
# MODEL MODULES (from models/modules.py)
# ================================================================================

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = F.conv2d(x, w, padding=w_pad+f_pad)
            x = F.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
            a = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ================================================================================
# DHARIWAL UNET (from models/unets.py)
# ================================================================================

class DhariwalUNet(nn.Module):
    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, augment_dim=0,
                 model_channels=192, channel_mult=[1,2,3,4], channel_mult_emb=4, num_blocks=3,
                 attn_resolutions=[32,16,8], dropout=0.10, label_dropout=0):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        self.enc = nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = F.silu(emb)

        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x


# ================================================================================
# EDM PRECONDITIONER (from models/precond.py)
# ================================================================================

class EDMPrecond(nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, model_type='DhariwalUNet', **model_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = DhariwalUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

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


# ================================================================================
# SCHEDULER (from utils/scheduler.py)
# ================================================================================

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
            [2 * scaling_fn(time_steps[i])**2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])
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


# ================================================================================
# NAVIER-STOKES SOLVER (from inverse_problems/navier_stokes.py)
# ================================================================================

class NavierStokes2d(object):
    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, Re=100.0, device=None, dtype=torch.float64):
        self.s1 = s1
        self.s2 = s2
        self.L1 = L1
        self.L2 = L2
        self.Re = Re
        self.h = 1.0/max(s1, s2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.zeros((1,)),
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        self.dealias = (k1**2 + k2**2 <= (s1/3)**2 + (s2/3)**2).type(dtype).to(device)
        self.dealias[0,0] = 0.0

    def stream_function(self, w_h, real_space=False):
        psi_h = self.inv_lap*w_h
        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    def velocity_field(self, stream_f, real_space=True):
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f
        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    def nonlinear_term(self, w_h, f_h=None):
        dealias_w_h = w_h*self.dealias
        w = fft.irfft2(dealias_w_h, s=(self.s1, self.s2))
        q, v = self.velocity_field(self.stream_function(dealias_w_h, real_space=False), real_space=True)
        nonlin = -1j*((2*math.pi/self.L1)*self.k1*fft.rfft2(q*w) + (2*math.pi/self.L1)*self.k2*fft.rfft2(v*w))
        if f_h is not None:
            nonlin += f_h
        return nonlin

    def time_step(self, q, v, f, Re):
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


# ================================================================================
# BASE OPERATOR (from inverse_problems/base.py)
# ================================================================================

class BaseOperator(ABC):
    def __init__(self, sigma_noise=0.0, unnorm_shift=0.0, unnorm_scale=1.0, device='cuda'):
        self.sigma_noise = sigma_noise
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

    @abstractmethod
    def forward(self, inputs, **kwargs):
        pass

    def __call__(self, inputs, **kwargs):
        target = inputs['target']
        out = self.forward(target, **kwargs)
        return out + self.sigma_noise * torch.randn_like(out)

    def gradient(self, pred, observation, return_loss=False):
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


# ================================================================================
# FORWARD NAVIER-STOKES 2D (from inverse_problems/navier_stokes.py)
# ================================================================================

class ForwardNavierStokes2d(BaseOperator):
    def __init__(self, resolution=128, L=2*math.pi, forward_time=1.0, Re=200.0, downsample_factor=2,
                 dtype=torch.float32, delta_t=1e-2, adaptive=True, **kwargs):
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
    def __call__(self, data, unnormalize=True):
        x = data['target']
        sol = self.forward(x, unnormalize)
        sol += self.sigma_noise * torch.randn_like(sol)
        return sol

    @torch.no_grad()
    def forward(self, x, unnormalize=True):
        if unnormalize:
            raw_u = self.unnormalize(x)
        else:
            raw_u = x
        sol = self.solver.solve(raw_u.squeeze(1), self.force, self.forward_time, self.Re, adaptive=self.adaptive, delta_t=self.delta_t)
        sol = sol[..., ::self.downsample_factor, ::self.downsample_factor]
        return sol.unsqueeze(1).to(torch.float32)


# ================================================================================
# LMDB DATASET (from training/dataset.py)
# ================================================================================

class LMDBData(Dataset):
    def __init__(self, root, resolution=128, raw_resolution=128, num_channels=1, norm=True, mean=0.0, std=5.0, id_list=None):
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
            import torchvision.transforms.functional as TF
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


# ================================================================================
# EVALUATOR (from eval.py)
# ================================================================================

def relative_l2(pred, target):
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2


class Evaluator(ABC):
    def __init__(self, metric_list, forward_op=None, data_misfit=False):
        self.metric_list = metric_list
        self.forward_op = forward_op
        self.data_misfit = data_misfit
        if data_misfit:
            assert forward_op is not None
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


# ================================================================================
# SCG ALGORITHM (from algo/scg.py)
# ================================================================================

class Algo(ABC):
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op

    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass


class SCG(Algo):
    def __init__(self, net, forward_op, diffusion_scheduler_config, num_candidates=8, threshold=0.25, batch_size=8):
        super(SCG, self).__init__(net, forward_op)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.num_candidates = num_candidates
        self.batch_size = batch_size
        self.threshold = threshold
        assert self.num_candidates % self.batch_size == 0, 'Number of candidates should be divisible by batch size.'

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=False):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        num_batches = self.num_candidates // self.batch_size
        num_steps = self.scheduler.num_steps
        pbar = tqdm(range(num_steps))
        x_results = torch.empty(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)

        for j in range(num_samples):
            x_next = x_initial[j:j+1]
            for i in pbar:
                x_cur = x_next
                sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
                denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
                score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                if i < int(num_steps * self.threshold):
                    x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * torch.randn_like(x_cur)
                elif i > int(num_steps * self.threshold) and i < num_steps - 1:
                    sigma_next = self.scheduler.sigma_steps[i+1]
                    epsilon = torch.randn(self.num_candidates, *x_cur.shape[1:], device=device)
                    x_candidates = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon

                    loss_ensemble = torch.zeros(self.num_candidates, device=device)
                    for k in range(num_batches):
                        x_batch = x_candidates[k*self.batch_size:(k+1)*self.batch_size]
                        denoised_batch = self.net(x_batch / self.scheduler.scaling_steps[i+1], torch.as_tensor(sigma_next).to(x_cur.device))
                        loss_ensemble[k*self.batch_size:(k+1)*self.batch_size] = self.forward_op.loss(denoised_batch, observation)
                    idx = torch.argmin(loss_ensemble)
                    x_next = x_candidates[idx:idx+1]
                    loss_scale = loss_ensemble[idx]
                    pbar.set_description(f'Iteration {i + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
                else:
                    x_next = denoised
            x_results[j] = x_next
        return x_results


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set TF32 precision
    if CONFIG["tf32"]:
        torch.set_float32_matmul_precision("high")

    # Set random seed
    torch.manual_seed(CONFIG["seed"])

    # Set up directory for logging and saving data (relative path)
    exp_dir = os.path.join(CONFIG["exp_dir"], CONFIG["algorithm_name"], CONFIG["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)

    # Save config
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # Instantiate forward operator
    forward_op = ForwardNavierStokes2d(
        resolution=CONFIG["problem"]["resolution"],
        L=2 * math.pi,
        forward_time=CONFIG["problem"]["forward_time"],
        Re=CONFIG["problem"]["Re"],
        downsample_factor=CONFIG["problem"]["downsample_factor"],
        sigma_noise=CONFIG["problem"]["sigma_noise"],
        unnorm_scale=CONFIG["problem"]["unnorm_scale"],
        adaptive=CONFIG["problem"]["adaptive"],
        delta_t=CONFIG["problem"]["delta_t"],
        device=device
    )

    # Instantiate dataset (relative path from JSON config)
    testset = LMDBData(
        root=CONFIG["data"]["root"],
        resolution=CONFIG["data"]["resolution"],
        std=CONFIG["data"]["std"],
        id_list=CONFIG["data"]["id_list"]
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    logger.info(f"Loaded {len(testset)} test samples...")

    # Load pre-trained model (relative path from JSON config)
    ckpt_path = CONFIG["problem"]["prior"]
    
    # Create model
    net = EDMPrecond(
        img_resolution=CONFIG["pretrain"]["img_resolution"],
        img_channels=CONFIG["pretrain"]["img_channels"],
        label_dim=CONFIG["pretrain"]["label_dim"],
        model_channels=CONFIG["pretrain"]["model_channels"],
        channel_mult=CONFIG["pretrain"]["channel_mult"],
        attn_resolutions=CONFIG["pretrain"]["attn_resolutions"],
        num_blocks=CONFIG["pretrain"]["num_blocks"],
        dropout=CONFIG["pretrain"]["dropout"]
    )

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema' in ckpt.keys():
        net.load_state_dict(ckpt['ema'])
    else:
        net.load_state_dict(ckpt['net'])
    net = net.to(device)
    del ckpt
    net.eval()

    logger.info(f"Loaded pre-trained model from {ckpt_path}...")

    # Set up algorithm
    algo = SCG(
        net=net,
        forward_op=forward_op,
        diffusion_scheduler_config=CONFIG["algorithm"]["diffusion_scheduler_config"],
        num_candidates=CONFIG["algorithm"]["num_candidates"],
        threshold=CONFIG["algorithm"]["threshold"],
        batch_size=CONFIG["algorithm"]["batch_size"]
    )

    # Set up evaluator
    evaluator = NavierStokes2dEvaluator(forward_op=forward_op)

    # Run inference on single sample only (as per speed-up requirement)
    for i, data in enumerate(testloader):
        if i >= 1:  # Only run on first sample
            break
        
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)

        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')

        if CONFIG["inference"]:
            # Get the observation
            observation = forward_op(data)
            target = data['target']
            
            # Run the algorithm
            logger.info(f'Running inference on test sample {data_id}...')
            recon = algo.inference(observation, num_samples=CONFIG["num_samples"])
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

            result_dict = {
                'observation': observation,
                'recon': forward_op.unnormalize(recon).cpu(),
                'target': forward_op.unnormalize(target).cpu(),
            }
            torch.save(result_dict, save_path)
            logger.info(f"Saved results to {save_path}.")
        else:
            # Load the results
            result_dict = torch.load(save_path)
            logger.info(f"Loaded results from {save_path}.")

        # Evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=result_dict['observation'])
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    # Aggregate the results
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")


if __name__ == "__main__":
    main()
