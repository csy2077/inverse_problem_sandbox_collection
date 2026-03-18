import os
import re
import io
import copy
import glob
import time
import pickle
import hashlib
import logging
import tempfile
import urllib
import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
import torchvision.transforms.functional as TF
from tqdm import tqdm
from piq import psnr, ssim
import lmdb
import pandas as pd
import ehtim as eh
import ehtim.statistics.dataframes as ehdf


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
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
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

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
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
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
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

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
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
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
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
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
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

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
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
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim           = 0,
        augment_dim         = 0,
        model_channels      = 128,
        channel_mult        = [1,2,2,2],
        channel_mult_emb    = 4,
        num_blocks          = 4,
        attn_resolutions    = [16],
        dropout             = 0.10,
        label_dropout       = 0,
        embedding_type      = 'positional',
        channel_mult_noise  = 1,
        encoder_type        = 'standard',
        decoder_type        = 'standard',
        resample_filter     = [1,1],
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
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
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = F.silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(F.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        beta_d          = 19.9,
        beta_min        = 0.1,
        M               = 1000,
        epsilon_t       = 1e-5,
        model_type      = 'SongUNet',
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        
        _model_dict = {'SongUNet': SongUNet}
        self.model = _model_dict[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

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
        pred_grad = grad(loss, pred_tmp)[0]
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

class BlackHoleImaging(BaseOperator):
    """
    PyTorch Version of Black Hole Imaging Forward Operator based on eht-imaging library.
    """

    def __init__(self, root='dataset/blackhole', observation_time_ratio=1.0, noise_type='vis_thermal', ttype='direct', imsize=64, w1=0,
                 w2=1, w3=1, w4=0.5, sigma_noise=0.0, unnorm_shift=1.0, unnorm_scale=0.5, ref_flux=None, device='cuda'):
        super().__init__(sigma_noise, unnorm_shift, unnorm_scale, device)
        A_vis, A_cp, A_camp, obs, im, multiplier, sigma = self.process_obs(root, imsize, observation_time_ratio)

        if ref_flux is not None:
            im.ivec *= ref_flux / np.sum(im.ivec)

        self.ref_im = im
        self.ref_flux = np.sum(self.ref_im.ivec)
        self.ref_obs = obs
        self.ref_multiplier = multiplier
        self.observation_time_ratio = observation_time_ratio
        self.noise_type = noise_type
        self.ttype = ttype
        self.device = device

        self.get_index_matrix(obs)

        self.sigma = torch.tensor(sigma).to(device)

        self.A_vis = torch.from_numpy(A_vis).unsqueeze(0).unsqueeze(0).cfloat().to(device)
        self.A_cp = torch.from_numpy(A_cp).unsqueeze(1).unsqueeze(1).cfloat().to(device)
        self.A_camp = torch.from_numpy(A_camp).unsqueeze(1).unsqueeze(1).cfloat().to(device)

        self.amp_dim = self.A_vis.shape[-2]
        self.cphase_dim = self.A_cp.shape[-2]
        self.logcamp_dim = self.A_camp.shape[-2]
        self.flux_dim = 1

        self.C = 1
        self.H = imsize
        self.W = imsize
        self.weight_amp = w1 * self.amp_dim
        self.weight_cp = w2 * self.cphase_dim
        self.weight_camp = w3 * self.logcamp_dim
        self.weight_flux = w4

    @staticmethod
    def process_obs(root, imsize, observation_time_ratio=1.0):
        obsfile = root + '/' + 'obs.uvfits'
        gtfile = root + '/' + 'gt.fits'
        obs = eh.obsdata.load_uvfits(obsfile)
        pd_data = pd.DataFrame(obs.data)
        time_list = np.array(sorted(list(set(pd_data['time']))))
        time_list = time_list[:int(len(time_list) * observation_time_ratio)]
        pd_data = pd_data[pd_data['time'].isin(time_list)]
        obsdata = pd_data.to_records(index=False).view(np.ndarray).astype(eh.DTPOL_STOKES)
        obs.data = obsdata
        im = eh.image.load_fits(gtfile)
        im = im.regrid_image(im.fovx(), imsize)
        multiplier = im.ivec.max()
        sigma = obs.data['sigma'] / multiplier
        _, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, im, mask=[])
        _, _, A_cp = eh.imaging.imager_utils.chisqdata_cphase(obs, im, mask=[])
        _, _, A_camp = eh.imaging.imager_utils.chisqdata_logcamp(obs, im, mask=[])
        return A_vis, np.stack(A_cp, axis=0), np.stack(A_camp, axis=0), obs, im, multiplier, sigma

    @staticmethod
    def estimate_flux(obs):
        data = obs.unpack_bl('ALMA', 'APEX', 'amp')
        amp_list = []
        for pair in data:
            amp = pair[0][1]
            amp_list.append(amp)
        flux = np.median(amp_list)
        return flux

    def get_index_matrix(self, obs):
        obs_data_df = pd.DataFrame(obs.data)
        map_fn, conjugate_fn = {}, {}
        for i, (time, t1, t2) in enumerate(zip(obs_data_df['time'], obs_data_df['t1'], obs_data_df['t2'])):
            map_fn[(time, t1, t2)] = i
            conjugate_fn[(time, t1, t2)] = 0
            map_fn[(time, t2, t1)] = i
            conjugate_fn[(time, t2, t1)] = 1

        bispec_df = pd.DataFrame(obs.bispectra(count='min'))
        cp_index, cp_conjugate = [], []
        for time, t1, t2, t3 in zip(bispec_df['time'], bispec_df['t1'], bispec_df['t2'], bispec_df['t3']):
            idx = [map_fn[(time, t1, t2)], map_fn[(time, t2, t3)], map_fn[(time, t3, t1)]]
            conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t2, t3)], conjugate_fn[(time, t3, t1)]]
            cp_index.append(idx)
            cp_conjugate.append(conj)
        self.cp_index = torch.tensor(cp_index).long().to(self.device)
        self.cp_conjugate = torch.tensor(cp_conjugate).long().to(self.device)

        camp_df = pd.DataFrame(obs.c_amplitudes(count='min'))
        camp_index, camp_conjugate = [], []
        for time, t1, t2, t3, t4 in zip(camp_df['time'], camp_df['t1'], camp_df['t2'], camp_df['t3'], camp_df['t4']):
            idx = [map_fn[(time, t1, t2)], map_fn[(time, t3, t4)], map_fn[(time, t1, t4)], map_fn[(time, t2, t3)]]
            conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t3, t4)], conjugate_fn[(time, t1, t4)],
                    conjugate_fn[(time, t2, t3)]]
            camp_index.append(idx)
            camp_conjugate.append(conj)
        self.camp_index = torch.tensor(camp_index).long().to(self.device)
        self.camp_conjugate = torch.tensor(camp_conjugate).long().to(self.device)

    def forward_vis(self, x):
        x = x.to(self.A_vis)
        xvec = x.reshape(-1, self.C, 1, self.H * self.W)
        vis = (self.A_vis * xvec).sum(-1, keepdims=True)
        return vis

    def forward_amp(self, x):
        amp = self.forward_vis(x).abs()
        sigmaamp = self.sigma[None, None, :, None] + 0 * amp
        return amp, sigmaamp

    def forward_flux(self, x):
        return x.flatten(1).sum(-1)[:, None, None, None]

    def forward_bisepectra_from_image(self, x):
        x = x.to(self.A_cp)
        xvec = x.reshape(-1, self.C, 1, self.H * self.W)
        i1 = (self.A_cp[0] * xvec).sum(-1, keepdims=True)
        i2 = (self.A_cp[1] * xvec).sum(-1, keepdims=True)
        i3 = (self.A_cp[2] * xvec).sum(-1, keepdims=True)
        return i1, i2, i3

    def forward_cp_from_image(self, x):
        i1, i2, i3 = self.forward_bisepectra_from_image(x)
        cphase = torch.angle(i1 * i2 * i3)
        v1 = self.sigma[self.cp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.cp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.cp_index[:, 2]][None, None, :, None]
        sigmacp = (v1 ** 2 / i1.abs() ** 2 + v2 ** 2 / i2.abs() ** 2 + v3 ** 2 / i3.abs() ** 2).sqrt()
        return cphase, sigmacp

    def forward_logcamp_bispectra_from_image(self, x):
        x = x.to(self.A_camp)
        x_vec = x.reshape(-1, self.C, 1, self.H * self.W)
        i1 = (self.A_camp[0] * x_vec).sum(-1, keepdims=True).abs()
        i2 = (self.A_camp[1] * x_vec).sum(-1, keepdims=True).abs()
        i3 = (self.A_camp[2] * x_vec).sum(-1, keepdims=True).abs()
        i4 = (self.A_camp[3] * x_vec).sum(-1, keepdims=True).abs()
        return i1, i2, i3, i4

    def forward_logcamp_from_image(self, x):
        i1, i2, i3, i4 = self.forward_logcamp_bispectra_from_image(x)
        camp = i1.log() + i2.log() - i3.log() - i4.log()
        v1 = self.sigma[self.camp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.camp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.camp_index[:, 2]][None, None, :, None]
        v4 = self.sigma[self.camp_index[:, 3]][None, None, :, None]
        sigmaca = (v1 ** 2 / i1 ** 2 + v2 ** 2 / i2 ** 2 + v3 ** 2 / i3 ** 2 + v4 ** 2 / i4 ** 2).sqrt()
        return camp, sigmaca

    def forward_from_image(self, x):
        amp, sigmaamp = self.forward_amp(x)
        cphase, sigmacp = self.forward_cp_from_image(x)
        logcamp, sigmacamp = self.forward_logcamp_from_image(x)
        flux = self.forward_flux(x)
        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    def correct_vis_direction(self, vis, conj):
        vis = vis * (1 - conj) + vis.conj() * conj
        return vis

    def forward_amp_from_vis(self, vis):
        amp = vis.abs()
        sigmaamp = self.sigma[None, None, :, None] + 0 * amp
        return amp, sigmaamp

    def forward_bisepectra_from_vis(self, vis):
        v1 = vis[:, :, self.cp_index[:, 0], :]
        v2 = vis[:, :, self.cp_index[:, 1], :]
        v3 = vis[:, :, self.cp_index[:, 2], :]
        cj1 = self.cp_conjugate[None, None, :, 0, None]
        cj2 = self.cp_conjugate[None, None, :, 1, None]
        cj3 = self.cp_conjugate[None, None, :, 2, None]
        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        return i1, i2, i3

    def forward_cp_from_vis(self, vis):
        i1, i2, i3 = self.forward_bisepectra_from_vis(vis)
        cphase = torch.angle(i1 * i2 * i3)
        v1 = self.sigma[self.cp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.cp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.cp_index[:, 2]][None, None, :, None]
        sigmacp = (v1 ** 2 / i1.abs() ** 2 + v2 ** 2 / i2.abs() ** 2 + v3 ** 2 / i3.abs() ** 2).sqrt()
        return cphase, sigmacp

    def forward_logcamp_bispectra_from_vis(self, vis):
        v1 = vis[:, :, self.camp_index[:, 0], :].abs()
        v2 = vis[:, :, self.camp_index[:, 1], :].abs()
        v3 = vis[:, :, self.camp_index[:, 2], :].abs()
        v4 = vis[:, :, self.camp_index[:, 3], :].abs()
        cj1 = self.camp_conjugate[None, None, :, 0, None]
        cj2 = self.camp_conjugate[None, None, :, 1, None]
        cj3 = self.camp_conjugate[None, None, :, 2, None]
        cj4 = self.camp_conjugate[None, None, :, 3, None]
        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        i4 = self.correct_vis_direction(v4, cj4)
        return i1, i2, i3, i4

    def forward_logcamp_from_vis(self, vis):
        i1, i2, i3, i4 = self.forward_logcamp_bispectra_from_vis(vis)
        logcamp = i1.log() + i2.log() - i3.log() - i4.log()
        v1 = self.sigma[self.camp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.camp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.camp_index[:, 2]][None, None, :, None]
        v4 = self.sigma[self.camp_index[:, 3]][None, None, :, None]
        sigmaca = (v1 ** 2 / i1 ** 2 + v2 ** 2 / i2 ** 2 + v3 ** 2 / i3 ** 2 + v4 ** 2 / i4 ** 2).sqrt()
        return logcamp, sigmaca

    def forward_from_vis(self, x):
        vis = self.forward_vis(x)
        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)
        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    @staticmethod
    def pt2ehtim(pt_image, res, ref_im):
        im = copy.deepcopy(ref_im)
        im.ivec = pt_image.clip(0, 1).detach().cpu().numpy().reshape(res, res).flatten()
        return im

    @staticmethod
    def pt2ehtim_batch(pt_images, res, ref_im):
        eht_images = []
        for pt_image in pt_images:
            eh_image = copy.deepcopy(ref_im)
            eh_image.ivec = pt_image.clip(0, 1).detach().cpu().numpy().reshape(res, res).flatten()
            eht_images.append(eh_image)
        return eht_images

    def measure_eht(self, x):
        ref_im = self.ref_im
        multiplier = self.ref_multiplier
        ref_obs = self.ref_obs
        res = self.H
        pt_obs = []
        for pt_image in x:
            eh_image = self.pt2ehtim(pt_image, res, ref_im)
            eh_image.ivec = eh_image.ivec * multiplier
            obs = eh_image.observe_same(ref_obs, phasecal=False, ampcal=False, ttype=self.ttype, verbose=False)
            adf = ehdf.make_amp(obs, debias=False)
            amp = torch.from_numpy(adf['amp'].to_numpy())[None, None, :, None].float().to(x.device) / multiplier
            sigmaamp = torch.from_numpy(adf['sigma'].to_numpy())[None, None, :, None].float().to(x.device) / multiplier
            cdf = ehdf.make_cphase_df(obs, count='min')
            cp = torch.from_numpy(cdf['cphase'].to_numpy())[None, None, :, None].float().to(x.device) * eh.DEGREE
            sigmacp = torch.from_numpy(cdf['sigmacp'].to_numpy())[None, None, :, None].float().to(x.device) * eh.DEGREE
            ldf = ehdf.make_camp_df(obs, count='min')
            camp = torch.from_numpy(ldf['camp'].to_numpy())[None, None, :, None].float().to(x.device)
            sigmaca = torch.from_numpy(ldf['sigmaca'].to_numpy())[None, None, :, None].float().to(x.device)
            flux = torch.tensor([self.estimate_flux(obs)])[None, None, :, None].float().to(x.device) / multiplier
            y = torch.cat([amp, sigmaamp, cp, sigmacp, camp, sigmaca, flux], dim=2)
            pt_obs.append(y)
        pt_obs = torch.cat(pt_obs, dim=0).to(x.device)
        return pt_obs

    def measure_guassian(self, x):
        vis = self.forward_vis(x)
        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)
        amp = amp + torch.randn_like(amp) * sigmaamp
        cphase = cphase + torch.randn_like(cphase) * sigmacp
        logcamp = logcamp + torch.randn_like(logcamp) * sigmacamp
        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    def measure_vis_error(self, x):
        vis = self.forward_vis(x)
        sigma = self.sigma[None, None, :, None].repeat(x.shape[0], 1, 1, 1)
        vis = vis + (torch.randn_like(vis) + 1j * torch.randn_like(vis)) * sigma
        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)
        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    def compress(self, amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux):
        return torch.cat([amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux], dim=2)

    def decompress(self, y):
        cur = 0
        amp = y[:, :, cur:cur + self.amp_dim]
        cur += self.amp_dim
        sigmaamp = y[:, :, cur:cur + self.amp_dim]
        cur += self.amp_dim
        cphase = y[:, :, cur:cur + self.cphase_dim]
        cur += self.cphase_dim
        sigmacp = y[:, :, cur:cur + self.cphase_dim]
        cur += self.cphase_dim
        logcamp = y[:, :, cur:cur + self.logcamp_dim]
        cur += self.logcamp_dim
        sigmacamp = y[:, :, cur:cur + self.logcamp_dim]
        cur += self.logcamp_dim
        flux = y[:, :, cur:cur + self.flux_dim]
        cur += self.flux_dim
        assert cur == y.shape[2]
        return amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux

    def chi2_amp(self, x, y_amp, y_amp_sigma):
        amp_pred, _ = self.forward_amp(x)
        return self.chi2_amp_from_meas(amp_pred, y_amp, y_amp_sigma)

    @staticmethod
    def chi2_amp_from_meas(y_amp_meas, y_amp, y_amp_sigma):
        residual = y_amp_meas - y_amp
        return torch.mean(torch.square(residual / y_amp_sigma), dim=(1, 2, 3))

    def chi2_cphase(self, x, y_cphase, y_cphase_sigma):
        cphase_pred, _ = self.forward_cp_from_image(x)
        return self.chi2_cphase_from_meas(cphase_pred, y_cphase, y_cphase_sigma)

    @staticmethod
    def chi2_cphase_from_meas(y_cphase_meas, y_cphase, y_cphase_sigma):
        angle_residual = y_cphase - y_cphase_meas
        return 2. * torch.mean((1 - torch.cos(angle_residual)) / torch.square(y_cphase_sigma), dim=(1, 2, 3))

    def chi2_logcamp(self, x, y_camp, y_logcamp_sigma):
        y_camp_pred, _ = self.forward_logcamp_from_image(x)
        return self.chi2_logcamp_from_meas(y_camp_pred, y_camp, y_logcamp_sigma)

    @staticmethod
    def chi2_logcamp_from_meas(y_logcamp_meas, y_logcamp, y_logcamp_sigma):
        return torch.mean(torch.abs((y_logcamp_meas - y_logcamp) / y_logcamp_sigma) ** 2, dim=(1, 2, 3))

    def chi2_flux(self, x, y_flux):
        flux_pred = self.forward_flux(x)
        return self.chi2_flux_from_meas(flux_pred, y_flux)

    @staticmethod
    def chi2_flux_from_meas(y_flux_meas, y_flux):
        return torch.mean(torch.square((y_flux_meas - y_flux) / 2), dim=(1, 2, 3))

    @staticmethod
    def normalize_chisq(chisq):
        overfit = chisq < 1.0
        e_chisq = chisq * (~overfit) + 1 / chisq * overfit
        return e_chisq

    def evaluate_chisq(self, x, y, normalize=True):
        _, _, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        x_flux = self.forward_flux(x)
        x_aligned = x * (y_flux / x_flux)
        cp_loss = self.chi2_cphase(x_aligned, y_cp, y_cphase_sigma)
        camp_loss = self.chi2_logcamp(x_aligned, y_camp, y_logcamp_sigma)
        if normalize:
            cp_loss = self.normalize_chisq(cp_loss)
            camp_loss = self.normalize_chisq(camp_loss)
        return cp_loss, camp_loss

    @staticmethod
    def aligned_images(image1, image2, search_range=(-0.5, 0.5), steps=30):
        batch_size, shape = image1.shape[0], image1.shape[1:]
        tx_values = torch.linspace(search_range[0], search_range[1], steps)
        ty_values = torch.linspace(search_range[0], search_range[1], steps)
        tx, ty = torch.meshgrid(tx_values, ty_values)
        tx, ty = tx.flatten(), ty.flatten()
        first_row = torch.stack([torch.ones_like(tx), torch.zeros_like(tx), tx], dim=-1)
        second_row = torch.stack([torch.zeros_like(ty), torch.ones_like(ty), ty], dim=-1)
        theta = torch.stack([first_row, second_row], dim=1)
        grid = F.affine_grid(theta, (tx.shape[0], *shape), align_corners=True)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        N, S = grid.shape[:2]
        flatten_image1 = image1.unsqueeze(1).repeat(1, S, 1, 1, 1).flatten(0, 1).clip(0, 1)
        flatten_image2 = image2.unsqueeze(1).repeat(1, S, 1, 1, 1).flatten(0, 1).clip(0, 1)
        flatten_grid = grid.flatten(0, 1)
        trans_image2 = F.grid_sample(flatten_image2.cpu(), flatten_grid.cpu(), align_corners=True).to(image1.device).clip(0, 1)
        eval_psnr = psnr(flatten_image1, trans_image2, data_range=1.0, reduction='none')
        argmax = eval_psnr.view(N, S).max(dim=1)[1]
        aligned_image2 = trans_image2.view(N, S, *image2.shape[1:])[torch.arange(N), argmax]
        return aligned_image2

    def blur_images(self, samples, factor=15):
        eht_images = self.pt2ehtim_batch(samples, 64, self.ref_im)
        blur_samples = []
        for eht_image in eht_images:
            blur_eht_image = eht_image.blur_circ(factor * eh.RADPERUAS)
            pt_image = torch.from_numpy(blur_eht_image.ivec.reshape(1, 1, 64, 64).astype(np.float32))
            blur_samples.append(pt_image)
        blur_samples = torch.cat(blur_samples).to(samples.device)
        return blur_samples

    @staticmethod
    def aligned_psnr(image1, aligned_image2):
        return psnr(image1.clip(0, 1), aligned_image2.clip(0, 1), data_range=1.0, reduction='none')

    def blur_aligned_psnr(self, image1, aligned_image2, factor=15):
        blur_image1 = self.blur_images(image1, factor).clip(0, 1)
        blur_aligned_image2 = self.blur_images(aligned_image2, factor).clip(0, 1)
        return psnr(blur_image1, blur_aligned_image2, data_range=1.0, reduction='none')

    def evaluate_psnr(self, image1, image2, blur_factors=(0, 10, 15, 20)):
        aligned_image2 = self.aligned_images(image1, image2)
        eval_psnr = self.aligned_psnr(image1, aligned_image2)[:, None]
        for f in blur_factors:
            f_psnr = self.blur_aligned_psnr(image1, aligned_image2, factor=f)[:, None]
            eval_psnr = torch.cat([eval_psnr, f_psnr], dim=1)
        return eval_psnr

    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale

    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift

    def forward(self, x, **kwargs):
        x = self.unnormalize(x)
        return self.forward_from_vis(x)

    def __call__(self, data, **kwargs):
        x = data['target']
        x = self.unnormalize(x)
        if self.noise_type == 'gaussian':
            return self.measure_guassian(x)
        elif self.noise_type == 'vis_thermal':
            return self.measure_vis_error(x)
        elif self.noise_type == 'eht':
            return self.measure_eht(x)
        else:
            raise ValueError('Unknown noise type')

    def loss(self, x, y):
        x = self.unnormalize(x)
        y_amp, y_amp_sigma, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        amp_loss = self.chi2_amp(x, y_amp, y_amp_sigma)
        cp_loss = self.chi2_cphase(x, y_cp, y_cphase_sigma)
        camp_loss = self.chi2_logcamp(x, y_camp, y_logcamp_sigma)
        flux_loss = self.chi2_flux(x, y_flux)
        data_fit = self.weight_amp * amp_loss + self.weight_cp * cp_loss + self.weight_camp * camp_loss + self.weight_flux * flux_loss
        return data_fit * 2

class BlackHole(Dataset):
    def __init__(self, root, resolution=64, original_resolution=400,
                 random_flip=True, zoom_in_out=True, zoom_range=[0.833, 1.145], id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.original_resolution = original_resolution
        self.length = self.txn.stat()['entries']
        self.random_flip = random_flip
        self.zoom_in_out = zoom_in_out
        self.zoom_range = zoom_range

        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    def __len__(self):
        return self.length

    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def __getitem__(self, idx):
        key = f'{self.idx_map(idx)}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float64).reshape(1, self.original_resolution, self.original_resolution)
        img = torch.from_numpy(np.array(img, copy=True))
        if self.zoom_in_out:
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoom_shape = [int(self.resolution * scale), int(self.resolution * scale)]
            img = TF.resize(img, zoom_shape, antialias=True)
            if zoom_shape[0] > self.resolution:
                img = TF.center_crop(img, self.resolution)
            elif zoom_shape[0] < self.resolution:
                diff = self.resolution - zoom_shape[0]
                img = TF.pad(img, (diff // 2 + diff % 2, diff // 2 + diff % 2, diff // 2, diff // 2))
        else:
            img = TF.resize(img, (self.resolution, self.resolution), antialias=True)

        img /= img.max()
        img = 2 * img - 1

        if self.random_flip and np.random.rand() < 0.5:
            img = torch.flip(img, [2])
        if self.random_flip and np.random.rand() < 0.5:
            img = torch.flip(img, [1])
        return {'target': img.float()}

class Algo(ABC):
    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op
    
    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass

class LGD(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples=10,
                 batch_grad=True,
                 sde=True):
        super(LGD, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                                self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], \
                self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            samples = denoised + torch.randn((self.num_samples, *denoised.shape[1:]), device=device) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(samples, observation, return_loss=True)
                gradients = gradient
                avg_loss = loss_scale
            else:
                gradients = torch.empty((self.num_samples, *denoised.shape[1:]), device=device)
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(samples[j:j+1], observation, return_loss=True)
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()

            avg_grad = torch.mean(gradients, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)
            
            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale):.4f}')

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale

        return x_next

class BlackHoleEvaluator:
    def __init__(self, forward_op=None):
        self.metric_list = {'cp_chi2': None, 'camp_chi2': None, 'psnr': None, 'blur_psnr (f=10)': None,
                           'blur_psnr (f=15)': None, 'blur_psnr (f=20)': None}
        self.forward_op = forward_op
        self.device = forward_op.device
        self.metric_state = {key: [] for key in self.metric_list.keys()}

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        pred, target, observation = pred.to(self.device), target.to(self.device), observation.to(self.device)

        if pred.shape != target.shape:
            target = target.repeat(pred.shape[0], 1, 1, 1)
            observation = observation.repeat(pred.shape[0], 1, 1, 1)

        chisq_cp, chisq_logcamp = self.forward_op.evaluate_chisq(pred, observation, True)

        blur_factors = [0, 10, 15, 20]
        blur_psnr = self.forward_op.evaluate_psnr(target, pred, blur_factors)
        blur_psnr = blur_psnr.max(dim=0)[0]

        metric_dict['cp_chi2'] = chisq_cp.min().item()
        metric_dict['camp_chi2'] = chisq_logcamp.min().item()
        metric_dict['psnr'] = blur_psnr[0].item()
        metric_dict['blur_psnr (f=10)'] = blur_psnr[1].item()
        metric_dict['blur_psnr (f=15)'] = blur_psnr[2].item()
        metric_dict['blur_psnr (f=20)'] = blur_psnr[3].item()

        self.metric_state['cp_chi2'].append(metric_dict['cp_chi2'])
        self.metric_state['camp_chi2'].append(metric_dict['camp_chi2'])
        self.metric_state['psnr'].append(metric_dict['psnr'])
        self.metric_state['blur_psnr (f=10)'].append(metric_dict['blur_psnr (f=10)'])
        self.metric_state['blur_psnr (f=15)'].append(metric_dict['blur_psnr (f=15)'])
        self.metric_state['blur_psnr (f=20)'].append(metric_dict['blur_psnr (f=20)'])
        return metric_dict

    def compute(self):
        metric_state = {}
        for key, val in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

def main():
    # Configuration (matching the command line args)
    config = EasyDict({
        'tf32': True,
        'inference': True,
        'num_samples': 1,
        'compile': False,
        'seed': 0,
        'wandb': False,
        'exp_name': 'standalone_lgd',
        'problem': EasyDict({
            'name': 'blackhole',
            'prior': 'weights/blackhole-50k.pt',
            'model': {
                'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/blackhole_test',
                'imsize': 64,
                'observation_time_ratio': 1.0,
                'noise_type': 'eht',
                'w1': 0,
                'w2': 1,
                'w3': 1,
                'w4': 0.5,
                'sigma_noise': 0.0,
                'unnorm_scale': 0.5,
                'unnorm_shift': 1.0
            },
            'data': {
                'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/blackhole_test',
                'resolution': 64,
                'original_resolution': 64,
                'random_flip': False,
                'zoom_in_out': False,
                'id_list': '0'  # Only first test sample
            },
            'exp_dir': 'exps/inference/blackhole'
        }),
        'algorithm': EasyDict({
            'name': 'LGD',
            'method': {
                'diffusion_scheduler_config': {
                    'num_steps': 1000,
                    'schedule': 'vp',
                    'timestep': 'vp',
                    'scaling': 'vp'
                },
                'guidance_scale': 0.003,  # From command line: algorithm.method.guidance_scale=0.003
                'num_samples': 3,
                'batch_grad': True,
                'sde': True
            }
        }),
        'pretrain': EasyDict({
            'model': {
                'model_type': 'SongUNet',
                'img_resolution': 64,
                'img_channels': 1,
                'label_dim': 0,
                'model_channels': 128,
                'channel_mult': [1, 2, 2],
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
    
    # Instantiate Problem Model (BlackHoleImaging)
    logger.info("Instantiating BlackHoleImaging problem model...")
    forward_op = BlackHoleImaging(
        **config.problem.model,
        device=device
    )

    # Instantiate Dataset (only first sample)
    logger.info("Instantiating BlackHole Dataset...")
    testset = BlackHole(
        root=config.problem.data['root'],
        resolution=config.problem.data['resolution'],
        original_resolution=config.problem.data['original_resolution'],
        random_flip=config.problem.data['random_flip'],
        zoom_in_out=config.problem.data['zoom_in_out'],
        id_list=config.problem.data['id_list']
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f"Loaded {len(testset)} test sample(s)...")

    # Load Pretrained Model
    logger.info("Loading pre-trained model...")
    ckpt_path = config.problem.prior
    
    # Handle relative paths
    if not os.path.exists(ckpt_path) and not is_url(ckpt_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(script_dir, ckpt_path)
    
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        logger.info(f"Could not load pickle, trying torch.load from {ckpt_path}")
        net = VPPrecond(**config.pretrain.model)
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
    
    # Instantiate Algorithm (LGD)
    logger.info("Instantiating LGD algorithm...")
    algo = LGD(
        net=net,
        forward_op=forward_op,
        **config.algorithm.method
    )

    # Instantiate Evaluator
    logger.info("Instantiating BlackHoleEvaluator...")
    evaluator = BlackHoleEvaluator(forward_op=forward_op)

    # Inference Loop (only first sample)
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

        # Evaluate the results
        metric_dict = evaluator(pred=result_dict['recon'].to(device), target=result_dict['target'].to(device), observation=observation.to(device) if isinstance(observation, torch.Tensor) else observation)
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
    
    forward_op.close()
