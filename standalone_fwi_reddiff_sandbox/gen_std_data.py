import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_
import inspect as _inspect_
import json as _json_
_META_REGISTRY_ = set()
try:
    import numpy as _np_
except ImportError:
    _np_ = None
try:
    import torch as _torch_
except ImportError:
    _torch_ = None

def _fix_seeds_(seed=42):
    import random
    if _np_:
        _np_.random.seed(seed)
    random.seed(seed)
    if _torch_:
        _torch_.manual_seed(seed)
        if _torch_.cuda.is_available():
            _torch_.cuda.manual_seed_all(seed)
_fix_seeds_(42)

def _analyze_obj_(obj):
    if _torch_ and isinstance(obj, _torch_.Tensor):
        return {'type': 'torch.Tensor', 'shape': list(obj.shape), 'dtype': str(obj.dtype), 'device': str(obj.device)}
    if _np_ and isinstance(obj, _np_.ndarray):
        return {'type': 'numpy.ndarray', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return {'type': type(obj).__name__, 'length': len(obj), 'elements': [_analyze_obj_(item) for item in obj]}
    if hasattr(obj, '__dict__'):
        methods = []
        try:
            for m in dir(obj):
                if m.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, m)
                    if callable(attr):
                        methods.append(m)
                except Exception:
                    continue
        except Exception:
            pass
        return {'type': 'CustomObject', 'class_name': obj.__class__.__name__, 'public_methods': methods, 'attributes': list(obj.__dict__.keys())}
    try:
        val_str = str(obj)
    except:
        val_str = '<non-stringifiable>'
    return {'type': type(obj).__name__, 'value_sample': val_str}

def _record_io_decorator_(save_path='./'):

    def decorator(func, parent_function=None):

        @_functools_.wraps(func)
        def wrapper(*args, **kwargs):
            global _META_REGISTRY_
            func_name = func.__name__
            parent_key = str(parent_function)
            registry_key = (func_name, parent_key)
            should_record = False
            if registry_key not in _META_REGISTRY_:
                should_record = True
            result = None
            inputs_meta = {}
            if should_record:
                try:
                    sig = _inspect_.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for (name, value) in bound_args.arguments.items():
                        inputs_meta[name] = _analyze_obj_(value)
                except Exception as e:
                    inputs_meta = {'error': f'Analysis failed: {e}'}
            result = func(*args, **kwargs)
            if should_record:
                try:
                    output_meta = _analyze_obj_(result)
                except Exception:
                    output_meta = 'Analysis failed'
                try:
                    final_path = save_path
                    if not final_path.endswith('.json'):
                        if not _os_.path.exists(final_path):
                            _os_.makedirs(final_path, exist_ok=True)
                        if parent_function == None:
                            final_path = _os_.path.join(final_path, f'IO_meta_{func_name}.json')
                        else:
                            final_path = _os_.path.join(final_path, f'IO_meta_parent_function_{parent_function}_{func_name}.json')
                    dir_name = _os_.path.dirname(final_path)
                    if dir_name and (not _os_.path.exists(dir_name)):
                        _os_.makedirs(dir_name, exist_ok=True)
                    existing_data = []
                    file_exists = _os_.path.exists(final_path)
                    if file_exists:
                        try:
                            with open(final_path, 'r') as f:
                                existing_data = _json_.load(f)
                        except:
                            pass
                    already_in_file = False
                    for entry in existing_data:
                        if entry.get('function_name') == func_name:
                            already_in_file = True
                            break
                    if not already_in_file:
                        func_schema = {'function_name': func_name, 'inputs': inputs_meta, 'output': output_meta}
                        existing_data.append(func_schema)
                        with open(final_path, 'w') as f:
                            _json_.dump(existing_data, f, indent=4)
                        print(f'  [Metadata] Recorded schema for: {func_name}')
                    _META_REGISTRY_.add(registry_key)
                except Exception as e:
                    print(f'  [Metadata] Warning: {e}')
            if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
                return decorator(result, parent_function=func_name)
            return result
        return wrapper
    return decorator

def _data_capture_decorator_(func, parent_function=None):

    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/std_data'
            if not _os_.path.exists(out_dir):
                _os_.makedirs(out_dir, exist_ok=True)
            func_name = func.__name__
            if parent_function == None:
                save_path = _os_.path.join(out_dir, f'data_{func_name}.pkl')
            else:
                save_path = _os_.path.join(out_dir, f'data_parent_{parent_function}_{func_name}.pkl')

            def detach_recursive(obj):
                if hasattr(obj, 'detach'):
                    return obj.detach()
                if isinstance(obj, list):
                    return [detach_recursive(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple((detach_recursive(x) for x in obj))
                if isinstance(obj, dict):
                    return {k: detach_recursive(v) for (k, v) in obj.items()}
                return obj
            payload = {'func_name': func_name, 'args': detach_recursive(args), 'kwargs': detach_recursive(kwargs), 'output': detach_recursive(result)}
            with open(save_path, 'wb') as f:
                _dill_.dump(payload, f)
        except Exception as e:
            pass
        if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
            return _data_capture_decorator_(result, parent_function=func_name)
        return result
    return wrapper
import os
import gc
import re
import io
import json
import math
import copy
import ctypes
import hashlib
import glob
import uuid
import tempfile
import urllib
import logging
from abc import ABC, abstractmethod
from functools import partial
from collections import OrderedDict
from typing import Any, Dict, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import lmdb
import tqdm
import torchvision.transforms.functional as TF
from devito import Function
from devito import configuration
from examples.seismic import Model, Receiver, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from distributed import Client, LocalCluster
from piq import psnr, ssim
configuration['log-level'] = 'WARNING'

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'data_standalone', 'standalone_fwi_reddiff.json')
    with open(config_path, 'r') as f:
        return json.load(f)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def trim_memory() -> int:
    libc = ctypes.CDLL('libc.so.6')
    return libc.malloc_trim(0)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def parse_int_list(s):
    if isinstance(s, list):
        return s
    if isinstance(s, int):
        return [s]
    ranges = []
    range_re = re.compile('^(\\d+)-(\\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def create_logger(logging_dir, main_process=True):
    """Create a logger that writes to a log file and stdout."""
    if not main_process:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[\x1b[34m%(asctime)s\x1b[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(f'{logging_dir}/log.txt')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
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

    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False, resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
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
        if self.fused_resample and self.up and (w is not None):
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and (w is not None):
            x = F.conv2d(x, w, padding=w_pad + f_pad)
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

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-05):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class UNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False, num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-05, resample_filter=[1, 1], resample_proj=False, adaptive_scale=True, init=dict(), init_zero=dict(init_weight=0), init_attn=None):
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
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1, **init_attn if init_attn is not None else init)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            (scale, shift) = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        if self.num_heads:
            (q, k, v) = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
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
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class DhariwalUNet(nn.Module):

    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, augment_dim=0, model_channels=192, channel_mult=[1, 2, 3, 4], channel_mult_emb=4, num_blocks=3, attn_resolutions=[32, 16, 8], dropout=0.1, label_dropout=0):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None
        self.enc = nn.ModuleDict()
        cout = in_channels
        for (level, mult) in enumerate(channel_mult):
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
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=res in attn_resolutions, **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]
        self.dec = nn.ModuleDict()
        for (level, mult) in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=res in attn_resolutions, **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

class EDMPrecond(nn.Module):

    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, model_type='DhariwalUNet', **model_kwargs):
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
        dtype = torch.float16 if self.use_fp16 and (not force_fp32) and (x.device.type == 'cuda') else torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

class Scheduler:

    def __init__(self, num_steps=10, sigma_max=100, sigma_min=0.01, sigma_final=None, schedule='linear', timestep='poly-7', scaling='none'):
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
        (sigma_fn, sigma_derivative_fn, sigma_inv_fn) = self.get_sigma_fn(self.schedule)
        time_step_fn = self.get_time_step_fn(self.timestep, self.sigma_max, self.sigma_min)
        (scaling_fn, scaling_derivative_fn) = self.get_scaling_fn(scaling)
        if self.schedule == 'vp':
            self.sigma_max = sigma_fn(1) * scaling_fn(1)
        time_steps = np.array([time_step_fn(s) for s in steps])
        time_steps = np.append(time_steps, sigma_inv_fn(self.sigma_final))
        sigma_steps = np.array([sigma_fn(t) for t in time_steps])
        scaling_steps = np.array([scaling_fn(t) for t in time_steps])
        scaling_factor = np.array([1 - scaling_derivative_fn(time_steps[i]) / scaling_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])
        factor_steps = np.array([2 * scaling_fn(time_steps[i]) ** 2 * sigma_fn(time_steps[i]) * sigma_derivative_fn(time_steps[i]) * (time_steps[i] - time_steps[i + 1]) for i in range(num_steps)])
        (self.sigma_steps, self.time_steps, self.factor_steps, self.scaling_factor, self.scaling_steps) = (sigma_steps, time_steps, factor_steps, scaling_factor, scaling_steps)
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
        return (sigma_fn, sigma_derivative_fn, sigma_inv_fn)

    def get_scaling_fn(self, schedule):
        if schedule == 'vp':
            beta_d = 19.9
            beta_min = 0.1
            scaling_fn = lambda t: 1 / np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t))
            scaling_derivative_fn = lambda t: -(beta_d * t + beta_min) / 2 / np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t))
        else:
            scaling_fn = lambda t: 1
            scaling_derivative_fn = lambda t: 0
        return (scaling_fn, scaling_derivative_fn)

    def get_time_step_fn(self, timestep, sigma_max, sigma_min):
        if timestep == 'log':
            get_time_step_fn = lambda r: sigma_max ** 2 * (sigma_min ** 2 / sigma_max ** 2) ** r
        elif timestep.startswith('poly'):
            p = int(timestep.split('-')[1])
            get_time_step_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p
        elif timestep == 'vp':
            get_time_step_fn = lambda r: 1 - r * (1 - 0.001)
        else:
            raise NotImplementedError
        return get_time_step_fn

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
        pred_grad = grad(loss, pred_tmp)[0]
        if return_loss:
            return (pred_grad, loss)
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

class fg_pair:

    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        return fg_pair(f, g)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def convert2np(rec):
    return np.array(rec.data)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def forward_single_shot(geometry, model, save=False, dt=1.0):
    solver_i = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs = solver_i.forward(vp=model.vp, save=save, dt=dt)[0]
    return d_obs.resample(dt)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def forward_multi_shots(model, geometry_list, client, save=False, dt=1.0, return_rec=True):
    forward_single_shot_fn = partial(forward_single_shot, model=model, save=save, dt=dt)
    futures = client.map(forward_single_shot_fn, geometry_list)
    if return_rec:
        shots = client.gather(futures)
        return shots
    else:
        shots_tmp = client.map(convert2np, futures)
        shots_np = client.gather(shots_tmp)
        return shots_np

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def gradient_single_shot(geometry, d_obs, model, fs=True):
    grad_devito = Function(name='grad', grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    (d_pred, u0) = solver.forward(vp=model.vp, save=True)[0:2]
    residual.data[:] = d_pred.data[:] - d_obs.resample(geometry.dt).data[:][0:d_pred.data.shape[0], :]
    fval = 0.5 * np.linalg.norm(residual.data.flatten()) ** 2
    solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad_devito)
    nbl = model.nbl
    z_start = 0 if fs else nbl
    grad_crop = np.array(grad_devito.data[:])[nbl:-nbl, z_start:-nbl]
    return fg_pair(fval, grad_crop)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def gradient_multi_shots(model, geometry_list, ob_recs, client, fs=True):
    gradient_single_shot_fn = partial(gradient_single_shot, model=model, fs=fs)
    fgi = client.map(gradient_single_shot_fn, geometry_list, ob_recs)
    fg = client.submit(sum, fgi).result()
    return (fg.f, fg.g)

class AcousticWave(BaseOperator):

    def __init__(self, shape, spacing, tn, f0, dt, nbl, nreceivers, nshots, src_depth=10.0, rec_depth=10.0, fs=True, bcs='damp', space_order=4, gc_threshold=1000, **kwargs):
        super().__init__(**kwargs)
        vel_init = np.ones(shape, dtype=np.float32)
        self.model = Model(vp=vel_init, origin=(0, 0), shape=shape, spacing=spacing, space_order=space_order, nbl=nbl, fs=fs, bcs=bcs, dt=dt)
        self.dt = dt
        self.fs = fs
        self.nshots = nshots
        self.nreceivers = nreceivers
        src_coordinates = np.empty((nshots, 2))
        src_coordinates[:, 0] = np.linspace(spacing[0], self.model.domain_size[0], num=nshots)
        src_coordinates[:, 1] = src_depth
        rec_coordinates = np.empty((nreceivers, 2))
        rec_coordinates[:, 0] = np.linspace(0, self.model.domain_size[0], num=nreceivers)
        rec_coordinates[:, 1] = rec_depth
        self.geometry_list = []
        self.solver_list = []
        for i in range(nshots):
            geometry_i = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates[i, :], 0.0, tn, f0=f0, src_type='Ricker')
            solver_i = AcousticWaveSolver(self.model, geometry_i, space_order=space_order)
            self.geometry_list.append(geometry_i)
            self.solver_list.append(solver_i)
        self.num_time_steps = geometry_i.time_axis.num
        print(f'Will record {self.num_time_steps} time steps.')
        cluster = LocalCluster(threads_per_worker=nshots, death_timeout=120)
        self.client = Client(cluster)
        self.client.run(gc.disable)
        self.num_calls = 0
        self.gc_threshold = gc_threshold

    def __call__(self, data: Dict[str, torch.Tensor], unnormalize=True):
        inputs = data['target']
        if unnormalize:
            inputs = self.unnormalize(inputs)
        vel_np = inputs.detach().transpose(-2, -1).cpu().numpy()[0, 0]
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        self.model.vp.data[nbl:-nbl, z_start:-nbl] = vel_np
        shots = forward_multi_shots(self.model, self.geometry_list, self.client, dt=self.dt, return_rec=True)
        return shots

    def forward(self, inputs, unnormalize=True):
        self.check_gc()
        if unnormalize:
            inputs = self.unnormalize(inputs)
        batch_vel_np = inputs.detach().transpose(-2, -1).cpu().numpy()
        out_np = np.empty((batch_vel_np.shape[0], self.nshots, self.num_time_steps, self.nreceivers), dtype=np.float32)
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        for i in range(batch_vel_np.shape[0]):
            self.model.vp.data[nbl:-nbl, z_start:-nbl] = batch_vel_np[i, 0]
            shots = forward_multi_shots(self.model, self.geometry_list, self.client, dt=self.dt, return_rec=False)
            shots_np = np.stack(shots, axis=0)
            out_np[i] = shots_np
        del shots_np
        out = torch.from_numpy(out_np).to(inputs.device)
        if torch.isnan(out).any():
            raise ValueError('NaN values in the forward evaluation.')
        return out

    def loss(self, pred, observation, unnormalize=True):
        self.check_gc()
        pred_out = self.forward(pred, unnormalize=unnormalize)
        obs_out = torch.from_numpy(np.stack([convert2np(obs) for obs in observation], axis=0)).to(pred.device)
        residual = pred_out - obs_out.unsqueeze(0)
        loss = 0.5 * torch.linalg.norm(residual.flatten(start_dim=1), dim=1) ** 2
        return loss

    def gradient(self, pred, observation, return_loss=False, unnormalize=True):
        self.check_gc()
        if unnormalize:
            pred = self.unnormalize(pred)
        pred_np = pred.detach().transpose(-2, -1).detach().cpu().numpy()
        nbl = self.model.nbl
        z_start = 0 if self.fs else nbl
        self.model.vp.data[nbl:-nbl, z_start:-nbl] = pred_np[0, 0]
        (fval, grad_slowness) = gradient_multi_shots(self.model, self.geometry_list, observation, self.client, fs=self.fs)
        if np.isnan(grad_slowness).any():
            raise ValueError('NaN values in the gradient.')
        if np.isnan(fval):
            raise ValueError('NaN values in the functional value.')
        grad_vel = -2.0 * grad_slowness / pred_np[0, 0] ** 3
        grad_vel = torch.from_numpy(grad_vel).transpose(0, 1).unsqueeze(0).unsqueeze(0).to(pred.device)
        if unnormalize:
            grad_vel = grad_vel * self.unnorm_scale
        if return_loss:
            return (grad_vel, torch.tensor(fval))
        else:
            return grad_vel

    def check_gc(self):
        if self.num_calls > self.gc_threshold:
            self.client.run(gc.collect)
            self.client.run(trim_memory)
            self.num_calls = 0
        else:
            self.num_calls += 1

    def close(self):
        self.client.close()

class REDDiff:

    def __init__(self, net, forward_op, num_steps=1000, observation_weight=1.0, base_lambda=0.25, base_lr=0.5, lambda_scheduling_type='constant'):
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp', timestep='vp', scaling='vp')
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        mu = torch.zeros(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))
        for step in pbar:
            with torch.no_grad():
                (sigma, scaling) = (self.scheduler.sigma_steps[step], self.scheduler.scaling_steps[step])
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()
            lam = self.lambda_fn(sigma)
            optimizer.zero_grad()
            (gradient, loss_scale) = self.forward_op.gradient(mu, observation, return_loss=True)
            gradient = gradient * self.observation_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient
            optimizer.step()
            pbar.set_description(f'Iteration {step + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
        return mu

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
            id_list = parse_int_list(id_list) if isinstance(id_list, str) else id_list
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def relative_l2(pred, target):
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2 = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm
    return rel_l2

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def fwi_norm(x):
    return (x - 1.5) / 3.0

class AcousticWaveEvaluator:

    def __init__(self, forward_op=None):
        self.metric_list = {'relative l2': relative_l2, 'psnr': lambda x, y: psnr(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none'), 'ssim': lambda x, y: ssim(fwi_norm(x).clip(0, 1), fwi_norm(y).clip(0, 1), data_range=1.0, reduction='none')}
        self.forward_op = forward_op
        self.device = forward_op.device if forward_op is not None else 'cpu'
        self.metric_state = {key: [] for key in self.metric_list.keys()}
        self.metric_state['data misfit'] = []

    def eval_data_misfit(self, pred, observation):
        data_misfit = self.forward_op.loss(pred, observation, unnormalize=False)
        return torch.sqrt(data_misfit)

    def __call__(self, pred, target, observation=None):
        metric_dict = {'data misfit': 0.0}
        for (metric_name, metric_func) in self.metric_list.items():
            if len(target.shape) == 3:
                val = metric_func(pred, target).item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
            else:
                val = metric_func(pred, target).mean().item()
                metric_dict[metric_name] = val
                self.metric_state[metric_name].append(val)
        data_misfit = self.eval_data_misfit(pred, observation).mean().item()
        metric_dict['data misfit'] = data_misfit
        self.metric_state['data misfit'].append(data_misfit)
        return metric_dict

    def compute(self):
        metric_state = {}
        for (key, val) in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_fwi_reddiff_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(config['seed'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(script_dir, config['exp_dir'], config['algorithm_name'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    forward_op = AcousticWave(shape=tuple(config['model']['shape']), spacing=tuple(config['model']['spacing']), tn=config['model']['tn'], f0=config['model']['f0'], dt=config['model']['dt'], nbl=config['model']['nbl'], nshots=config['model']['nshots'], nreceivers=config['model']['nreceivers'], unnorm_scale=config['model']['unnorm_scale'], unnorm_shift=config['model']['unnorm_shift'], src_depth=config['model']['src_depth'], device=device)
    data_root = os.path.join(script_dir, config['data']['root'])
    testset = LMDBData(root=data_root, resolution=config['data']['resolution'], raw_resolution=config['data']['raw_resolution'], std=config['data']['std'], mean=config['data']['mean'], id_list=config['data']['id_list'])
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f'Loaded {len(testset)} test samples...')
    ckpt_path = os.path.join(script_dir, config['prior'])
    net = EDMPrecond(img_resolution=config['pretrain']['img_resolution'], img_channels=config['pretrain']['img_channels'], label_dim=config['pretrain']['label_dim'], model_channels=config['pretrain']['model_channels'], channel_mult=config['pretrain']['channel_mult'], attn_resolutions=config['pretrain']['attn_resolutions'], num_blocks=config['pretrain']['num_blocks'], dropout=config['pretrain']['dropout'])
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'ema' in ckpt.keys():
        net.load_state_dict(ckpt['ema'])
    else:
        net.load_state_dict(ckpt['net'])
    net = net.to(device)
    del ckpt
    net.eval()
    logger.info(f'Loaded pre-trained model from {ckpt_path}...')
    algo = REDDiff(net=net, forward_op=forward_op, num_steps=config['algorithm']['num_steps'], observation_weight=config['algorithm']['observation_weight'], base_lr=config['algorithm']['base_lr'], base_lambda=config['algorithm']['base_lambda'], lambda_scheduling_type=config['algorithm']['lambda_scheduling_type'])
    evaluator = AcousticWaveEvaluator(forward_op=forward_op)
    for (i, data) in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for (key, val) in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if config['inference']:
            observation = forward_op(data)
            target = data['target']
            logger.info(f'Running inference on test sample {data_id}...')
            recon = algo.inference(observation, num_samples=config['num_samples'])
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
            result_dict = {'observation': observation, 'recon': forward_op.unnormalize(recon).cpu(), 'target': forward_op.unnormalize(target).cpu()}
            torch.save(result_dict, save_path)
            logger.info(f'Saved results to {save_path}.')
        else:
            result_dict = torch.load(save_path)
            logger.info(f'Loaded results from {save_path}.')
        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'], observation=result_dict['observation'])
        logger.info(f'Metric results: {metric_dict}...')
    logger.info('Evaluation completed...')
    metric_state = evaluator.compute()
    logger.info(f'Final metric results: {metric_state}...')
    forward_op.close()
if __name__ == '__main__':
    main()