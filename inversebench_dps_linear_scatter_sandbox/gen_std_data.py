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
            if callable(result) and (not isinstance(result, type)):
                return decorator(result, parent_function=func_name)
            return result
        return wrapper
    return decorator

def _data_capture_decorator_(func, parent_function=None):

    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data'
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
        if callable(result) and (not isinstance(result, type)):
            return _data_capture_decorator_(result, parent_function=func_name)
        return result
    return wrapper
import os
import sys
import re
import io
import math
import copy
import glob
import uuid
import time
import pickle
import hashlib
import logging
import tempfile
import urllib
import requests
import html
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from scipy.special import hankel1
from scipy.integrate import dblquad
from PIL import Image
import lmdb
from tqdm import tqdm
from piq import psnr, ssim, SSIMLoss
_cache_dir = 'cache'

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def set_cache_dir(path: str) -> None:
    global _cache_dir
    _cache_dir = path

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def is_url(obj: Any, allow_file_urls: bool=False) -> bool:
    if not isinstance(obj, str) or not '://' in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or (not '.' in res.netloc):
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, '/'))
        if not res.scheme or not res.netloc or (not '.' in res.netloc):
            return False
    except:
        return False
    return True

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def open_url(url: str, cache_dir: str=None, num_attempts: int=10, verbose: bool=True, return_filename: bool=False, cache: bool=True) -> Any:
    assert num_attempts >= 1
    assert not (return_filename and (not cache))
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, 'rb')
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match('^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, 'rb')
    assert is_url(url)
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')
    url_md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + '_*'))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, 'rb')
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print('Downloading %s ...' % url, end='', flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError('No data received')
                    if len(res.content) < 8192:
                        content_str = res.content.decode('utf-8')
                        if 'download_warning' in res.headers.get('Set-Cookie', ''):
                            links = [html.unescape(link) for link in content_str.split('"') if 'export=download' in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError('Google Drive virus checker nag')
                        if 'Google Drive - Quota exceeded' in content_str:
                            raise IOError('Google Drive download quota exceeded -- please try again later')
                    match = re.search('filename="([^"]*)"', res.headers.get('Content-Disposition', ''))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(' done')
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(' failed')
                    raise
                if verbose:
                    print('.', end='', flush=True)
    if cache:
        safe_name = re.sub('[^0-9a-zA-Z-._]', '_', url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + '_' + safe_name)
        temp_file = os.path.join(cache_dir, 'tmp_' + uuid.uuid4().hex + '_' + url_md5 + '_' + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, 'wb') as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)
        if return_filename:
            return cache_file
    assert not return_filename
    return io.BytesIO(url_data)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def create_logger(logging_dir, main_process=True):
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
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

class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class Conv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False, resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
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
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and (w is not None):
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class GroupNorm(torch.nn.Module):

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-05):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class UNetBlock(torch.nn.Module):

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
        x = self.conv0(F.silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            (scale, shift) = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale
        if self.num_heads:
            (q, k, v) = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
            a = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

class PositionalEmbedding(torch.nn.Module):

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

class DhariwalUNet(torch.nn.Module):

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
        self.enc = torch.nn.ModuleDict()
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
        self.dec = torch.nn.ModuleDict()
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
_model_dict = {'DhariwalUNet': DhariwalUNet}

class EDMPrecond(torch.nn.Module):

    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, model_type='DhariwalUNet', **model_kwargs):
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
        dtype = torch.float16 if self.use_fp16 and (not force_fp32) and (x.device.type == 'cuda') else torch.float32
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def full_propagate_to_sensor(f, utot_dom_set, sensor_greens_function_set, dx, dy):
    num_trans = utot_dom_set.shape[2]
    num_rec = sensor_greens_function_set.shape[2]
    contSrc = f[0, 0].unsqueeze(-1) * utot_dom_set
    conjSrc = torch.conj(contSrc).reshape(-1, num_trans)
    sensor_greens_func = sensor_greens_function_set.reshape(-1, num_rec)
    uscat_pred_set = dx * dy * torch.matmul(conjSrc.T, sensor_greens_func)
    return uscat_pred_set

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def generate_em_functions(p):
    (XPix, YPix) = np.meshgrid(p['x'], p['y'])
    hank_fun = lambda x: 1j * 0.25 * hankel1(0, x)
    transmitter_angles = np.linspace(0, 359, p['numTrans']) * np.pi / 180
    x_transmit = p['sensorRadius'] * np.cos(transmitter_angles)
    y_transmit = p['sensorRadius'] * np.sin(transmitter_angles)
    receiver_angles = np.linspace(0, 359, p['numRec']) * np.pi / 180
    x_receive = p['sensorRadius'] * np.cos(receiver_angles)
    y_receive = p['sensorRadius'] * np.sin(receiver_angles)
    p['receiverMask'] = np.ones((p['numTrans'], p['numRec']))
    diff_x_rp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numRec'])) - np.tile(x_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_rp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numRec'])) - np.tile(y_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_rec_to_pix = np.sqrt(diff_x_rp ** 2 + diff_y_rp ** 2)
    diff_x_tp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - np.tile(x_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_tp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - np.tile(y_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_trans_to_pix = np.sqrt(diff_x_tp ** 2 + diff_y_tp ** 2)
    p['uincDom'] = hank_fun(p['kb'] * distance_trans_to_pix)
    sensor_greens_function = hank_fun(p['kb'] * distance_rec_to_pix)
    p['sensorGreensFunction'] = p['kb'] ** 2 * sensor_greens_function
    x_green = np.arange(-p['Nx'], p['Nx']) * p['dx']
    y_green = np.arange(-p['Ny'], p['Ny']) * p['dy']
    (XGreen, YGreen) = np.meshgrid(x_green, y_green)
    R = np.sqrt(XGreen ** 2 + YGreen ** 2)
    domain_greens_function = hank_fun(p['kb'] * R)

    def integrand_real(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x ** 2 + y ** 2)).real)

    def integrand_imag(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x ** 2 + y ** 2)).imag)
    Ny = p['Ny']
    Nx = p['Nx']
    dx = p['dx']
    dy = p['dy']
    domain_greens_function[Ny, Nx] = dblquad(integrand_real, -dx / 2, dx / 2, -dy / 2, dy / 2)[0] / (dx * dy)
    domain_greens_function[Ny, Nx] += dblquad(integrand_imag, -dx / 2, dx / 2, -dy / 2, dy / 2)[0] / (dx * dy) * 1j
    p['domainGreensFunction'] = p['kb'] ** 2 * domain_greens_function
    return p

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def construct_parameters(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=360, numTrans=60, sensorRadius=1.6, device='cuda'):
    em = {}
    em['Lx'] = Lx
    em['Ly'] = Ly
    em['Nx'] = Nx
    em['Ny'] = Ny
    em['dx'] = em['Lx'] / em['Nx']
    em['dy'] = em['Ly'] / em['Ny']
    em['x'] = np.linspace(-em['Nx'] / 2, em['Nx'] / 2 - 1, em['Nx']) * em['dx']
    em['y'] = np.linspace(-em['Ny'] / 2, em['Ny'] / 2 - 1, em['Ny']) * em['dy']
    em['c'] = 299792458
    em['lambda'] = em['dx'] * wave
    em['freq'] = em['c'] / em['lambda'] / 1000000000.0
    em['numRec'] = numRec
    em['numTrans'] = numTrans
    em['sensorRadius'] = sensorRadius
    em['kb'] = 2 * np.pi / em['lambda']
    em = generate_em_functions(em)
    return (torch.from_numpy(em['domainGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['sensorGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['uincDom']).to(device).unsqueeze(-1), torch.from_numpy(em['receiverMask']).unsqueeze(-1))

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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def load_and_preprocess_data(config: dict, device: torch.device) -> dict:
    """
    Loads dataset, model, and creates forward operator parameters.
    Returns all data needed for inversion.
    """
    model_config = config['problem']['model']
    Lx = model_config['Lx']
    Ly = model_config['Ly']
    Nx = model_config['Nx']
    Ny = model_config['Ny']
    wave = model_config['wave']
    numRec = model_config['numRec']
    numTrans = model_config['numTrans']
    sensorRadius = model_config['sensorRadius']
    sigma_noise = model_config['sigma_noise']
    unnorm_shift = model_config['unnorm_shift']
    unnorm_scale = model_config['unnorm_scale']
    dx = Lx / Nx
    dy = Ly / Ny
    (domain_greens_function_set, sensor_greens_function_set, uinc_dom_set, receiver_mask_set) = construct_parameters(Lx, Ly, Nx, Ny, wave, numRec, numTrans, sensorRadius, device)
    sensor_greens_function_set = sensor_greens_function_set.to(torch.complex128)
    uinc_dom_set = uinc_dom_set.to(torch.complex128)
    forward_op_params = {'dx': dx, 'dy': dy, 'Nx': Nx, 'Ny': Ny, 'numRec': numRec, 'numTrans': numTrans, 'sigma_noise': sigma_noise, 'unnorm_shift': unnorm_shift, 'unnorm_scale': unnorm_scale, 'sensor_greens_function_set': sensor_greens_function_set, 'uinc_dom_set': uinc_dom_set, 'device': device}
    data_config = config['problem']['data']
    testset = LMDBData(root=data_config['root'], resolution=data_config['resolution'], mean=data_config['mean'], std=data_config['std'], id_list=data_config['id_list'])
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    ckpt_path = config['problem']['prior']
    pretrain_config = config['pretrain']['model']
    if not os.path.exists(ckpt_path) and (not is_url(ckpt_path)):
        if os.path.exists(os.path.join('InverseBench', ckpt_path)):
            ckpt_path = os.path.join('InverseBench', ckpt_path)
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        net = EDMPrecond(**pretrain_config)
        if is_url(ckpt_path):
            ckpt_file = open_url(ckpt_path, return_filename=True)
            ckpt = torch.load(ckpt_file, map_location=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        elif 'net' in ckpt.keys():
            net.load_state_dict(ckpt['net'])
        else:
            net.load_state_dict(ckpt)
        net = net.to(device)
    net.eval()
    scheduler_config = config['algorithm']['method']['diffusion_scheduler_config']
    guidance_scale = config['algorithm']['method']['guidance_scale']
    sde = config['algorithm']['method']['sde']
    return {'testloader': testloader, 'testset': testset, 'net': net, 'forward_op_params': forward_op_params, 'scheduler_config': scheduler_config, 'guidance_scale': guidance_scale, 'sde': sde, 'num_samples': config['num_samples'], 'device': device, 'exp_dir': config['problem']['exp_dir'], 'exp_name': config['exp_name'], 'algorithm_name': config['algorithm']['name'], 'inference': config['inference']}

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def forward_operator(x: torch.Tensor, forward_op_params: dict, unnormalize: bool=True) -> torch.Tensor:
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
    uscat_pred_set = full_propagate_to_sensor(f, uinc_dom_set[..., 0], sensor_greens_function_set[..., 0], dx, dy)
    return uscat_pred_set.unsqueeze(0)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def run_inversion(observation: torch.Tensor, net: torch.nn.Module, forward_op_params: dict, scheduler_config: dict, guidance_scale: float, sde: bool, num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Runs the DPS inversion algorithm using the forward_operator.
    Returns the reconstructed image.
    """
    scheduler = Scheduler(**scheduler_config)
    if num_samples > 1:
        observation = observation.repeat(num_samples, 1, 1, 1)
    x_initial = torch.randn(num_samples, net.img_channels, net.img_resolution, net.img_resolution, device=device) * scheduler.sigma_max
    x_next = x_initial
    x_next.requires_grad = True
    pbar = tqdm(range(scheduler.num_steps))
    for i in pbar:
        x_cur = x_next.detach().requires_grad_(True)
        sigma = scheduler.sigma_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        denoised = net(x_cur / scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
        pred_tmp = denoised.clone().detach().requires_grad_(True)
        y_pred = forward_operator(pred_tmp, forward_op_params, unnormalize=True)
        diff = y_pred - observation
        squared_diff = diff * diff.conj()
        loss = torch.sum(squared_diff.real)
        gradient = torch.autograd.grad(loss, pred_tmp)[0]
        loss_scale = loss.detach()
        ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
        ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)
        score = (denoised - x_cur / scheduler.scaling_steps[i]) / sigma ** 2 / scheduler.scaling_steps[i]
        pbar.set_description(f'Iteration {i + 1}/{scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale):.4f}')
        if sde:
            epsilon = torch.randn_like(x_cur)
            x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
        else:
            x_next = x_cur * scaling_factor + factor * score * 0.5
        x_next = x_next - ll_grad * guidance_scale
    return x_next

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def evaluate_results(recon: torch.Tensor, target: torch.Tensor, observation: torch.Tensor, forward_op_params: dict, logger: logging.Logger) -> dict:
    """
    Evaluates reconstruction quality using PSNR and SSIM metrics.
    """
    unnorm_shift = forward_op_params['unnorm_shift']
    unnorm_scale = forward_op_params['unnorm_scale']
    recon_unnorm = (recon + unnorm_shift) * unnorm_scale
    target_unnorm = (target + unnorm_shift) * unnorm_scale
    recon_clipped = recon_unnorm.clip(0, 1)
    target_clipped = target_unnorm.clip(0, 1)
    if recon_clipped.shape != target_clipped.shape:
        target_clipped = target_clipped.repeat(recon_clipped.shape[0], 1, 1, 1)
    psnr_val = psnr(recon_clipped, target_clipped, data_range=1.0, reduction='none').mean().item()
    ssim_val = ssim(recon_clipped, target_clipped, data_range=1.0, reduction='none').mean().item()
    metric_dict = {'psnr': psnr_val, 'ssim': ssim_val}
    logger.info(f'Metric results: {metric_dict}')
    return metric_dict
if __name__ == '__main__':
    config = {'tf32': True, 'inference': True, 'num_samples': 1, 'compile': False, 'seed': 0, 'exp_name': 'default', 'problem': {'name': 'inverse-scatter-linear', 'prior': 'weights/inv-scatter-5m.pt', 'model': {'Lx': 0.18, 'Ly': 0.18, 'Nx': 128, 'Ny': 128, 'wave': 6, 'numRec': 360, 'numTrans': 20, 'sensorRadius': 1.6, 'sigma_noise': 0.0001, 'unnorm_shift': 1.0, 'unnorm_scale': 0.5}, 'data': {'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/inv-scatter-test', 'resolution': 128, 'mean': 0.5, 'std': 0.25, 'id_list': '0'}, 'exp_dir': 'exps/inference/inv-scatter-linear'}, 'algorithm': {'name': 'DPS', 'method': {'diffusion_scheduler_config': {'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'}, 'guidance_scale': 10.0, 'sde': True}}, 'pretrain': {'model': {'model_type': 'DhariwalUNet', 'img_resolution': 128, 'img_channels': 1, 'label_dim': 0, 'model_channels': 128, 'channel_mult': [1, 1, 1, 2, 2], 'attn_resolutions': [16], 'num_blocks': 1, 'dropout': 0.0}}}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['tf32']:
        torch.set_float32_matmul_precision('high')
    torch.manual_seed(config['seed'])
    exp_dir = os.path.join(config['problem']['exp_dir'], config['algorithm']['name'], config['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    logger.info('Loading and preprocessing data...')
    data_dict = load_and_preprocess_data(config, device)
    testloader = data_dict['testloader']
    testset = data_dict['testset']
    net = data_dict['net']
    forward_op_params = data_dict['forward_op_params']
    scheduler_config = data_dict['scheduler_config']
    guidance_scale = data_dict['guidance_scale']
    sde = data_dict['sde']
    num_samples = data_dict['num_samples']
    do_inference = data_dict['inference']
    logger.info(f'Loaded {len(testset)} test samples...')
    all_metrics = {'psnr': [], 'ssim': []}
    for (i, data) in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = {'target': data.to(device)}
        elif isinstance(data, dict):
            for (key, val) in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if do_inference:
            target = data['target']
            sigma_noise = forward_op_params['sigma_noise']
            observation = forward_operator(target, forward_op_params, unnormalize=True)
            observation = observation + sigma_noise * torch.randn_like(observation.real) + 1j * sigma_noise * torch.randn_like(observation.real)
            logger.info(f'Running inference on test sample {data_id}...')
            recon = run_inversion(observation=observation, net=net, forward_op_params=forward_op_params, scheduler_config=scheduler_config, guidance_scale=guidance_scale, sde=sde, num_samples=num_samples, device=device)
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
            unnorm_shift = forward_op_params['unnorm_shift']
            unnorm_scale = forward_op_params['unnorm_scale']
            result_dict = {'observation': observation.cpu(), 'recon': ((recon + unnorm_shift) * unnorm_scale).cpu(), 'target': ((target + unnorm_shift) * unnorm_scale).cpu()}
            torch.save(result_dict, save_path)
            logger.info(f'Saved results to {save_path}.')
        elif os.path.exists(save_path):
            result_dict = torch.load(save_path)
            logger.info(f'Loaded results from {save_path}.')
            observation = result_dict['observation'].to(device)
            recon = result_dict['recon'].to(device)
            target = result_dict['target'].to(device)
            unnorm_shift = forward_op_params['unnorm_shift']
            unnorm_scale = forward_op_params['unnorm_scale']
            recon = recon / unnorm_scale - unnorm_shift
            target = target / unnorm_scale - unnorm_shift
        else:
            logger.warning(f'Result file {save_path} not found, skipping evaluation for this sample.')
            continue
        metric_dict = evaluate_results(recon=recon, target=data['target'] if do_inference else target, observation=observation, forward_op_params=forward_op_params, logger=logger)
        all_metrics['psnr'].append(metric_dict['psnr'])
        all_metrics['ssim'].append(metric_dict['ssim'])
    logger.info('Evaluation completed...')
    final_metrics = {'psnr': np.mean(all_metrics['psnr']), 'psnr_std': np.std(all_metrics['psnr']), 'ssim': np.mean(all_metrics['ssim']), 'ssim_std': np.std(all_metrics['ssim'])}
    logger.info(f'Final metric results: {final_metrics}...')
    print('OPTIMIZATION_FINISHED_SUCCESSFULLY')