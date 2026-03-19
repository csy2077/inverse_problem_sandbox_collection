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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/std_data'
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
import re
import logging
from typing import Any, Dict
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from piq import psnr, SSIMLoss
import lmdb
import sigpy as sp
from sigpy.mri import app

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def parse_int_list(s):
    if s is None:
        return None
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/meta_data.json')
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

class MultiCoilMRI(BaseOperator):

    def __init__(self, total_lines=128, acceleration_ratio=8, pattern='random', orientation='vertical', mask_fname=None, mask_seed=0, device='cuda', sigma_noise=0.0):
        """
        MRI forward operator
        Args:
            - total_lines: total number of phase-encoding lines
            - acceleration_ratio: undersampling factor
            - pattern: sampling pattern ('random' or 'equispaced')
            - orientation: orientation of phase encoding ('vertical' or 'horizontal')
            - mask_fname: path to mask file (if None, generate mask)
            - mask_seed: random seed for mask generation
            - device: device to run the operator
            - sigma_noise: noise standard deviation
        """
        super(MultiCoilMRI, self).__init__(sigma_noise=sigma_noise, device=device)
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
        """Phase encode random mask generator"""
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
        mask = np.zeros(total_lines)
        mask[center_line_idx] = 1.0
        mask[random_line_idx] = 1.0
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
        self.estimated_mvue = self.get_mvue_static(self.masked_kspace.cpu().numpy(), self.maps.cpu().numpy())
        return torch.view_as_real(self.masked_kspace)

    @staticmethod
    def get_mvue_static(kspace, s_maps):
        """Get mvue estimate from coil measurements"""
        return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image).to(torch.float64)
        coils = self.maps * torch.view_as_complex(image.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
        return torch.view_as_real(self.mask * self.fft(coils))

class MultiCoilMRIData:
    """Utility class for MRI data processing"""

    @staticmethod
    def get_mvue(kspace, s_maps):
        """Get mvue estimate from coil measurements"""
        return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

    @staticmethod
    def normalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img / scaling

    @staticmethod
    def unnormalize(gen_img, estimated_mvue):
        scaling = np.quantile(np.abs(estimated_mvue), 0.99)
        return gen_img * scaling

class MultiCoilMRILMDBData(Dataset):

    def __init__(self, root, image_size, mvue_only=False, slice_range=[5, -5], id_list=None, simulated_kspace=False):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.mvue_only = mvue_only
        self.simulated_kspace = simulated_kspace
        if not mvue_only:
            kspace_env = lmdb.open(str(self.root / 'kspace'), readonly=True, lock=False, create=False)
            s_maps_env = lmdb.open(str(self.root / 's_maps'), readonly=True, lock=False, create=False)
            self.kspace_txn = kspace_env.begin(write=False)
            self.s_maps_txn = s_maps_env.begin(write=False)
        mvue_env = lmdb.open(str(self.root / 'mvue'), readonly=True, lock=False, create=False)
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
            return {'target': torch.view_as_real(torch.from_numpy(mvue_scaled).squeeze(0)).permute(2, 0, 1).contiguous(), 'mvue': mvue_scaled, 'maps': maps, 'kspace': gt_ksp_scaled}

class Algo(ABC):

    def __init__(self, net, forward_op):
        self.net = net
        self.forward_op = forward_op

    @abstractmethod
    def inference(self, observation, num_samples=1, **kwargs):
        pass

class CompressedSensingMRI(Algo):
    """
    Compressed Sensing MRI algorithm using sigpy.
    
    Supports three modes:
    - 'Sense': SENSE reconstruction
    - 'L1Wavelet': L1 Wavelet regularized reconstruction  
    - 'TV': Total Variation regularized reconstruction
    """

    def __init__(self, net, forward_op, mode, lamda):
        super(CompressedSensingMRI, self).__init__(net, forward_op)
        self.mode = mode
        self.lamda = lamda

    @torch.no_grad()
    def inference(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        observation = torch.view_as_complex(observation)
        recon = torch.zeros(observation.shape[0], observation.shape[2], observation.shape[3], dtype=torch.complex128).to(self.forward_op.device)
        pbar = tqdm(range(len(observation)), desc=f'Running CS-MRI ({self.mode})')
        for i in pbar:
            masked_kspace = observation[i].cpu().numpy()
            s_maps = self.forward_op.maps[i].cpu().numpy()
            if self.mode == 'Sense':
                recon_app = app.SenseRecon(masked_kspace, s_maps, self.lamda)
            elif self.mode == 'L1Wavelet':
                recon_app = app.L1WaveletRecon(masked_kspace, s_maps, self.lamda)
            elif self.mode == 'TV':
                recon_app = app.TotalVariationRecon(masked_kspace, s_maps, self.lamda)
            else:
                raise ValueError(f'Invalid mode: {self.mode}. Choose from Sense, L1Wavelet, TV')
            recon[i] = torch.from_numpy(recon_app.run())
        return self.forward_op.normalize(torch.view_as_real(recon).permute(0, 3, 1, 2).contiguous())

class DynamicRangePSNRLoss:

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:
            yhats = torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        if ys.shape[1] == 2:
            ys = torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        return -torch.mean(torch.stack([psnr(yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0), data_range=y.max()) for (yhat, y) in zip(yhats, ys)]))

class DynamicRangeSSIMLoss:

    def __init__(self):
        self.ssim_loss = SSIMLoss()

    def __call__(self, yhats, ys):
        if yhats.shape[1] == 2:
            yhats = torch.view_as_complex(yhats.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        if ys.shape[1] == 2:
            ys = torch.view_as_complex(ys.permute(0, 2, 3, 1).contiguous()).type(torch.complex128).abs().unsqueeze(1)
        return torch.mean(torch.stack([SSIMLoss(data_range=y.max())(yhat.clip(0, y.max()).unsqueeze(0), y.unsqueeze(0)) for (yhat, y) in zip(yhats, ys)]))

class MRIEvaluator:

    def __init__(self, forward_op=None):
        dr_psnr_loss = DynamicRangePSNRLoss()
        dr_ssim_loss = DynamicRangeSSIMLoss()
        self.eval_batch = 32
        self.metric_list = {'psnr': lambda x, y: -dr_psnr_loss(x, y), 'ssim': lambda x, y: 1 - dr_ssim_loss(x, y)}
        self.forward_op = forward_op
        self.metric_state = defaultdict(list)
        self.device = forward_op.device if forward_op is not None else 'cpu'

    def __call__(self, pred, target, observation=None):
        """
        Args:
            - pred (torch.Tensor): (N, C, H, W)
            - target (torch.Tensor): (C, H, W) or (N, C, H, W)
        Returns:
            - metric_dict (Dict): a dictionary of metric values
        """
        metric_dict = {}
        for (metric_name, metric_func) in self.metric_list.items():
            metric_dict[metric_name] = 0.0
            if len(pred) != len(target):
                num_batches = pred.shape[0] // self.eval_batch
                for i in range(num_batches):
                    pred_batch = pred[i * self.eval_batch:(i + 1) * self.eval_batch]
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
        for (key, val) in self.metric_state.items():
            metric_state[key] = np.mean(val)
            metric_state[f'{key}_std'] = np.std(val)
        return metric_state

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mri_L1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    config = EasyDict({'tf32': True, 'inference': True, 'num_samples': 1, 'seed': 0, 'wandb': False, 'exp_name': 'standalone_first_sample', 'problem': EasyDict({'name': 'multi-coil-mri', 'prior': 'weights/MRI-knee.pt', 'model': {'sigma_noise': 0.0, 'total_lines': 320, 'acceleration_ratio': 4, 'pattern': 'random', 'mask_seed': 0}, 'data': {'root': '/fs-computility-new/UPDZ02_sunhe/chensiyi.p/data_downloads/knee_val_lmdb', 'image_size': [320, 320], 'id_list': '0', 'simulated_kspace': False}, 'evaluator': {}, 'exp_dir': 'exps/inference/multi-coil-mri-knee-standalone'}), 'algorithm': EasyDict({'name': 'CS-MRI-L1Wavelet', 'method': {'mode': 'L1Wavelet', 'lamda': 1e-06}})})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if config.tf32:
        torch.set_float32_matmul_precision('high')
    torch.manual_seed(config.seed)
    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    logger.info('Instantiating MultiCoilMRI problem model...')
    forward_op = MultiCoilMRI(**config.problem.model, device=device)
    logger.info('Instantiating Dataset...')
    testset = MultiCoilMRILMDBData(root=config.problem.data['root'], image_size=config.problem.data['image_size'], id_list=config.problem.data['id_list'], simulated_kspace=config.problem.data['simulated_kspace'])
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    logger.info(f'Loaded {len(testset)} test samples...')
    logger.info('Instantiating CS-MRI algorithm...')
    algo = CompressedSensingMRI(net=None, forward_op=forward_op, **config.algorithm.method)
    logger.info('Instantiating Evaluator...')
    evaluator = MRIEvaluator(forward_op=forward_op)
    for (i, data) in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for (key, val) in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
                elif isinstance(val, np.ndarray):
                    data[key] = torch.from_numpy(val).to(device) if val.dtype != np.complex64 and val.dtype != np.complex128 else val
        data_id = testset.id_list[i]
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if config.inference:
            observation = forward_op(data)
            target = data['target']
            logger.info(f'Running inference on test sample {data_id}...')
            recon = algo.inference(observation, num_samples=config.num_samples)
            recon = recon.to(device)
            logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')
            result_dict = {'observation': observation.cpu(), 'recon': forward_op.unnormalize(recon).cpu(), 'target': forward_op.unnormalize(target).cpu()}
            torch.save(result_dict, save_path)
            logger.info(f'Saved results to {save_path}.')
        elif os.path.exists(save_path):
            result_dict = torch.load(save_path)
            logger.info(f'Loaded results from {save_path}.')
            observation = result_dict['observation'].to(device)
        else:
            logger.warning(f'Result file {save_path} not found, skipping evaluation for this sample.')
            continue
        metric_dict = evaluator(pred=result_dict['recon'].to(device), target=result_dict['target'].to(device), observation=observation)
        logger.info(f'Metric results: {metric_dict}...')
    logger.info('Evaluation completed...')
    metric_state = evaluator.compute()
    logger.info(f'Final metric results: {metric_state}...')
    forward_op.close()
if __name__ == '__main__':
    main()