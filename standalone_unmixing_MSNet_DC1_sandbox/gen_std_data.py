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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MSNet_DC1_sandbox/run_code/std_data'
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
import json
import logging
import time
from math import ceil
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from munkres import Munkres
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log = logging.getLogger(__name__)
EPS = 1e-10
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class AdditiveWhiteGaussianNoise:

    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
        """
        Compute sigmas for the desired SNR given a flattened input HSI Y
        """
        log.debug(f'Y shape => {Y.shape}')
        assert len(Y.shape) == 2
        (L, N) = Y.shape
        log.info(f'Desired SNR => {self.SNR}')
        if self.SNR is None:
            sigmas = np.zeros(L)
        else:
            assert self.SNR > 0, 'SNR must be strictly positive'
            sigmas = np.ones(L)
            sigmas /= np.linalg.norm(sigmas)
            log.debug(f'Sigmas after normalization: {sigmas[0]}')
            num = np.sum(Y ** 2) / N
            denom = 10 ** (self.SNR / 10)
            sigmas_mean = np.sqrt(num / denom)
            log.debug(f'Sigma mean based on SNR: {sigmas_mean}')
            sigmas *= sigmas_mean
            log.debug(f'Final sigmas value: {sigmas[0]}')
        noise = np.diag(sigmas) @ np.random.randn(L, N)
        return Y + noise
INTEGER_VALUES = ('H', 'W', 'M', 'L', 'p', 'N')

class HSI:

    def __init__(self, dataset: str, data_dir: str='./data', figs_dir: str='./figs') -> None:
        self.H = 0
        self.W = 0
        self.M = 0
        self.L = 0
        self.p = 0
        self.N = 0
        self.Y = np.zeros((self.L, self.N))
        self.E = np.zeros((self.L, self.p))
        self.A = np.zeros((self.p, self.N))
        self.D = np.zeros((self.L, self.M))
        self.labels = []
        self.index = []
        self.name = dataset
        filename = f'{self.name}.mat'
        path = os.path.join(data_dir, filename)
        log.debug(f'Path to be opened: {path}')
        assert os.path.isfile(path)
        data = sio.loadmat(path)
        log.debug(f'Data keys: {data.keys()}')
        for key in filter(lambda k: not k.startswith('__'), data.keys()):
            self.__setattr__(key, data[key].item() if key in INTEGER_VALUES else data[key])
        if 'N' not in data.keys():
            self.N = self.H * self.W
        assert self.N == self.H * self.W
        assert self.Y.shape == (self.L, self.N)
        self.has_dict = False
        if 'D' in data.keys():
            self.has_dict = True
            assert self.D.shape == (self.L, self.M)
        if 'index' in data.keys():
            self.index = list(self.index.squeeze())
        self.figs_dir = figs_dir
        if self.figs_dir is not None:
            os.makedirs(self.figs_dir, exist_ok=True)

    def get_data(self):
        return (self.Y, self.p, self.D)

    def get_HSI_dimensions(self):
        return {'bands': self.L, 'pixels': self.N, 'lines': self.H, 'samples': self.W, 'atoms': self.M}

    def get_img_shape(self):
        return (self.H, self.W)

    def get_labels(self):
        return self.labels

    def get_index(self):
        return self.index

    def __repr__(self) -> str:
        msg = f'HSI => {self.name}\n'
        msg += '------------------------------\n'
        msg += f'{self.L} bands,\n'
        msg += f'{self.H} lines, {self.W} samples ({self.N} pixels),\n'
        msg += f'{self.p} endmembers ({self.labels}),\n'
        msg += f'{self.M} atoms\n'
        msg += f'GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n'
        return msg

    def plot_endmembers(self, E0=None, run=0):
        """
        Display endmembers
        """
        title = f'{self.name} - endmembers' + ' (GT)' if E0 is None else ''
        ylabel = 'Reflectance'
        xlabel = '# Bands'
        E = np.copy(self.E) if E0 is None else np.copy(E0)
        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            plt.plot(E[:, pp], label=self.labels[pp])
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        suffix = '-GT' if E0 is None else f'-{run}'
        figname = f'{self.name}-endmembers{suffix}.png'
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

    def plot_abundances(self, A0=None, run=0):
        (nrows, ncols) = (1, self.p)
        title = f'{self.name} - abundances' + ' (GT)' if A0 is None else ''
        A = np.copy(self.A) if A0 is None else np.copy(A0)
        A = A.reshape(self.p, self.H, self.W)
        (fig, ax) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(A[kk], vmin=0.0, vmax=1.0)
                curr_ax.set_title(f'{self.labels[kk]}')
                curr_ax.axis('off')
                fig.colorbar(mappable, ax=curr_ax, location='right', shrink=0.5)
                kk += 1
                if kk == self.p:
                    break
        plt.suptitle(title)
        suffix = '-GT' if A0 is None else f'-{run}'
        figname = f'{self.name}-abundances{suffix}.png'
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

class HSIWithGT(HSI):

    def __init__(self, dataset, data_dir, figs_dir):
        super().__init__(dataset=dataset, data_dir=data_dir, figs_dir=figs_dir)
        assert self.E.shape == (self.L, self.p)
        assert self.A.shape == (self.p, self.N)
        try:
            assert len(self.labels) == self.p
            tmp_labels = list(self.labels)
            self.labels = [s.strip(' ') for s in tmp_labels]
        except Exception:
            self.labels = [f'#{ii}' for ii in range(self.p)]
        assert np.allclose(self.A.sum(0), np.ones(self.N), rtol=0.001, atol=0.001)
        assert np.all(self.A >= -EPS)
        assert np.all(self.E >= -EPS)

    def get_GT(self):
        return (self.E, self.A)

    def has_GT(self):
        return True

class BaseMetric:

    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return (X, Xref)

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.name}'

class MSE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        (E, Eref) = self._check_input(E, Eref)
        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)
        return np.sqrt(normE.T ** 2 + normEref ** 2 - 2 * (E.T @ Eref))

class SpectralAngleDistance(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        (E, Eref) = self._check_input(E, Eref)
        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)
        tmp = (E / normE).T @ (Eref / normEref)
        ret = np.minimum(tmp, 1.0)
        return np.arccos(ret)

class SADDegrees(SpectralAngleDistance):

    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        tmp = super().__call__(E, Eref)
        return (np.diag(tmp) * (180 / np.pi)).mean()

class aRMSE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        (A, Aref) = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())

class eRMSE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        (E, Eref) = self._check_input(E, Eref)
        return 100 * np.sqrt(((E - Eref) ** 2).mean())

class SRE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        (X, Xref) = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, 'fro') / LA.norm(Xref - X, 'fro'))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MSNet_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """
    Return individual and global metric
    """
    d = {}
    d['Overall'] = round(metric(X_hat, X_gt), 4)
    if detail:
        for (ii, label) in enumerate(labels):
            if on_endmembers:
                (x_gt, x_hat) = (X_gt[:, ii][:, None], X_hat[:, ii][:, None])
                d[label] = round(metric(x_hat, x_gt), 4)
            else:
                d[label] = round(metric(X_hat[ii], X_gt[ii]), 4)
    log.info(f'{metric} => {d}')
    return d

class BaseAligner:

    def __init__(self, Aref, criterion):
        self.Aref = Aref
        self.criterion = criterion
        self.P = None
        self.dists = None

    def fit(self, A):
        raise NotImplementedError

    def transform(self, A):
        assert self.P is not None, 'Must be fitted first'
        assert A.shape[0] == self.P.shape[0]
        assert A.shape[0] == self.P.shape[1]
        return self.P @ A

    def transform_endmembers(self, E):
        assert self.P is not None, 'Must be fitted first'
        assert E.shape[1] == self.P.shape[0]
        assert E.shape[1] == self.P.shape[1]
        return E @ self.P.T

    def fit_transform(self, A):
        self.fit(A)
        res = self.transform(A)
        return res

    def __repr__(self):
        msg = f'{self.__class__.__name__}_crit{self.criterion}'
        return msg

class HungarianAligner(BaseAligner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, A):
        self.dists = self.criterion(A.T, self.Aref.T)
        p = A.shape[0]
        P = np.zeros((p, p))
        m = Munkres()
        indices = m.compute(self.dists)
        for (row, col) in indices:
            P[row, col] = 1.0
        self.P = P.T

class AbundancesAligner(HungarianAligner):

    def __init__(self, **kwargs):
        super().__init__(criterion=MSE(), **kwargs)

class VCA:

    def __init__(self):
        self.seed = None
        self.indices = None

    def extract_endmembers(self, Y, p, seed=0, snr_input=0, *args, **kwargs):
        """
        Vertex Component Analysis

        This code is a translation of a matlab code provided by
        Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        available at http://www.lx.it.pt/~bioucas/code.htm
        under a non-specified Copyright (c)
        Translation of last version at 22-February-2018 
        (Matlab version 2.1 (7-May-2004))
        """
        (L, N) = Y.shape
        self.seed = seed
        generator = np.random.default_rng(seed=self.seed)
        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :p]
            x_p = np.dot(Ud.T, Y_o)
            SNR = self.estimate_snr(Y, y_m, x_p)
            log.info(f'SNR estimated = {SNR}[dB]')
        else:
            SNR = snr_input
            log.info(f'input SNR = {SNR}[dB]\n')
        SNR_th = 15 + 10 * np.log10(p)
        if SNR < SNR_th:
            log.info('... Select proj. to R-1')
            d = p - 1
            if snr_input == 0:
                Ud = Ud[:, :d]
            else:
                y_m = np.mean(Y, axis=1, keepdims=True)
                Y_o = Y - y_m
                Ud = LA.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :d]
                x_p = np.dot(Ud.T, Y_o)
            Yp = np.dot(Ud, x_p[:d, :]) + y_m
            x = x_p[:d, :]
            c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            log.info('... Select the projective proj.')
            d = p
            Ud = LA.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]
            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(Ud, x_p[:d, :])
            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)
            y = x / np.dot(u.T, x)
        indices = np.zeros(p, dtype=int)
        A = np.zeros((p, p))
        A[-1, 0] = 1
        for i in range(p):
            w = generator.random(size=(p, 1))
            f = w - np.dot(A, np.dot(LA.pinv(A), w))
            f = f / np.linalg.norm(f)
            v = np.dot(f.T, y)
            indices[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indices[i]]
        E = Yp[:, indices]
        log.debug(f'Indices chosen to be the most pure: {indices}')
        self.indices = indices
        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        (L, N) = Y.shape
        (p, N) = x.shape
        P_y = np.sum(Y ** 2) / float(N)
        P_x = np.sum(x ** 2) / float(N) + np.sum(r_m ** 2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))
        return snr_est

class UnmixingModel:

    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f'{self.__class__.__name__}'
        return msg

    def print_time(self):
        return f'{self} took {self.time:.2f}s'

class BlindUnmixingModel(UnmixingModel):

    def __init__(self):
        super().__init__()

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):
        raise NotImplementedError(f'Solver is not implemented for {self}')

class MSNet(nn.Module, BlindUnmixingModel):

    def __init__(self, epochs=800, alpha=0.1, beta=0.03, drop_out=0.2, learning_rate=0.03, weight_decay=0.0001, step_size=30, gamma=0.6, *args, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def edge_loss(self, x, y):
        x = self.down22(x)
        x_laplace = self.laplacian_kernel(x)
        y_laplace = self.laplacian_kernel(y)
        diff = x_laplace - y_laplace
        return torch.sqrt(diff ** 2 + self.eps ** 2).mean()

    def init_architecture(self, seed):
        torch.manual_seed(seed)
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(self.p, 1, 1, 1)
        self.kernel = kernel.to(self.device)
        self.down22 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.eps = 0.001
        self.layer1 = nn.Sequential(nn.Conv2d(self.L + self.p, 96, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(96), nn.Dropout(self.drop_out), nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(48), nn.Dropout(self.drop_out), nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1))
        self.downsampling22 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.downsampling44 = nn.AvgPool2d(4, 4, ceil_mode=True)
        self.layer2 = nn.Sequential(nn.Conv2d(self.L + self.p, 96, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(96), nn.Dropout(self.drop_out), nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(48), nn.Dropout(self.drop_out), nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1))
        self.layer3 = nn.Sequential(nn.Conv2d(self.L, 96, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(96), nn.Dropout(self.drop_out), nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(48), nn.Dropout(self.drop_out), nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1))
        self.softmax = nn.Softmax(dim=1)
        self.transconv = nn.Conv2d(self.p, self.p, kernel_size=1, stride=1)
        self.transconv2 = nn.Conv2d(self.L, self.L, kernel_size=1, stride=1)
        self.decoderlayer4 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)
        self.decoderlayer5 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)
        self.decoderlayer6 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)

    def forward(self, x):
        down44 = self.downsampling44(x)
        layer3out = self.layer3(down44)
        en_result3 = self.softmax(layer3out)
        de_result3 = self.decoderlayer6(en_result3)
        translayer3 = F.interpolate(layer3out, (ceil(self.H / 2), ceil(self.W / 2)), mode='bilinear')
        translayer3 = self.transconv(translayer3)
        down22 = self.downsampling22(x)
        convlayer2 = self.transconv2(down22)
        layer2in = torch.cat((convlayer2, translayer3), 1)
        layer2out = self.layer2(layer2in)
        en_result2 = self.softmax(layer2out)
        de_result2 = self.decoderlayer5(en_result2)
        translayer2 = F.interpolate(layer2out, (self.H, self.W), mode='bilinear')
        translayer2 = self.transconv(translayer2)
        convlayer1 = self.transconv2(x)
        layer1in = torch.cat((convlayer1, translayer2), 1)
        layer1out = self.layer1(layer1in)
        en_result1 = self.softmax(layer1out)
        de_result1 = self.decoderlayer4(en_result1)
        return (en_result1, de_result1, en_result2, de_result2, en_result3, de_result3, down22, down44)

    @staticmethod
    def reconstruction_SAD_loss(output, target):
        assert output.shape == target.shape
        (_, band, h, w) = output.shape
        output = torch.reshape(output, (band, h * w))
        target = torch.reshape(target, (band, h * w))
        return torch.acos(torch.cosine_similarity(output, target, dim=0)).mean()

    def conv_gauss(self, img):
        (n_channels, _, kw, kh) = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def compute_endmembers_and_abundances(self, Y, p, H, W, seed=0, *args, **kwargs):
        tic = time.time()
        log.debug('Solving started...')
        (L, N) = Y.shape
        self.L = L
        self.p = p
        self.H = H
        self.W = W
        self.N = N
        self.init_architecture(seed=seed)
        extractor = VCA()
        Ehat = extractor.extract_endmembers(Y, p, seed=seed)
        Einit = torch.Tensor(Ehat).unsqueeze(2).unsqueeze(3)
        self.decoderlayer4.weight.data = Einit
        self.decoderlayer5.weight.data = Einit
        self.decoderlayer6.weight.data = Einit
        (num_channels, h, w) = (self.L, self.H, self.W)
        Y = torch.Tensor(Y)
        Y = Y.view(1, num_channels, h, w)
        self = self.to(self.device)
        Y = Y.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        progress = tqdm(range(self.epochs))
        self.train()
        for ii in progress:
            (en_abund, reconst_res, en_abund2, reconst_res2, en_abund3, reconst_res3, Y2, Y3) = self(Y)
            sad1 = self.reconstruction_SAD_loss(Y, reconst_res)
            sad2 = self.reconstruction_SAD_loss(Y2, reconst_res2)
            sad3 = self.reconstruction_SAD_loss(Y3, reconst_res3)
            A = sad1 + sad2 + sad3
            mse1 = F.mse_loss(Y, reconst_res)
            mse2 = F.mse_loss(Y2, reconst_res2)
            mse3 = F.mse_loss(Y3, reconst_res3)
            B = mse1 + mse2 + mse3
            edge1 = self.edge_loss(en_abund2, en_abund3)
            edge2 = self.edge_loss(en_abund, en_abund2)
            C = edge1 + edge2
            loss = A + self.alpha * B + self.beta * C
            progress.set_postfix_str(f'loss={loss.item():.3e}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.eval()
        (en_abund, reconst_res, en_abund2, reconst_res2, en_abund3, reconst_res3, Y2, Y3) = self(Y)
        self.time = time.time() - tic
        log.info(self.print_time())
        Ahat = en_abund.squeeze(0).reshape(self.p, self.N).detach().cpu().numpy()
        Ehat = self.decoderlayer4.weight.data.squeeze(-1).squeeze(-1).detach().cpu().numpy()
        return (Ehat, Ahat)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_MSNet_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    log.info('Blind Unmixing - MSNet on DC1 - [START]')
    config_path = os.path.join(SCRIPT_DIR, 'data_standalone', 'standalone_unmixing_MSNet_DC1.json')
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    seed = cfg['global']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = os.path.join(SCRIPT_DIR, 'exp_standalone_unmixing_MSNet_DC1')
    os.makedirs(output_dir, exist_ok=True)
    noise = AdditiveWhiteGaussianNoise(SNR=cfg['noise']['SNR'])
    data_dir = os.path.join(SCRIPT_DIR, 'data_standalone')
    hsi = HSIWithGT(dataset=cfg['data']['dataset'], data_dir=data_dir, figs_dir=output_dir)
    log.info(hsi)
    (Y, p, _) = hsi.get_data()
    (H, W) = hsi.get_img_shape()
    Y = noise.apply(Y)
    if cfg['global']['l2_normalization']:
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    model = MSNet(epochs=cfg['model']['epochs'], alpha=cfg['model']['alpha'], beta=cfg['model']['beta'], drop_out=cfg['model']['drop_out'], learning_rate=cfg['model']['learning_rate'], weight_decay=cfg['model']['weight_decay'], step_size=cfg['model']['step_size'], gamma=cfg['model']['gamma'])
    (E_hat, A_hat) = model.compute_endmembers_and_abundances(Y, p, H=H, W=W)
    estimates_path = os.path.join(output_dir, 'estimates.mat')
    sio.savemat(estimates_path, {'E': E_hat, 'A': A_hat.reshape(-1, H, W)})
    log.info(f'Estimates saved to {estimates_path}')
    metrics = {}
    if hsi.has_GT():
        (E_gt, A_gt) = hsi.get_GT()
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)
        labels = hsi.get_labels()
        metrics['SRE'] = compute_metric(SRE(), A_gt, A1, labels, detail=False, on_endmembers=False)
        metrics['aRMSE'] = compute_metric(aRMSE(), A_gt, A1, labels, detail=True, on_endmembers=False)
        metrics['SAD'] = compute_metric(SADDegrees(), E_gt, E1, labels, detail=True, on_endmembers=True)
        metrics['eRMSE'] = compute_metric(eRMSE(), E_gt, E1, labels, detail=True, on_endmembers=True)
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        log.info(f'Metrics saved to {metrics_path}')
        hsi.plot_endmembers(E0=E1, run=0)
        log.info(f'Endmember plot saved')
        hsi.plot_abundances(A0=A1, run=0)
        log.info(f'Abundance plot saved')
    log.info('Blind Unmixing - MSNet on DC1 - [END]')
if __name__ == '__main__':
    main()