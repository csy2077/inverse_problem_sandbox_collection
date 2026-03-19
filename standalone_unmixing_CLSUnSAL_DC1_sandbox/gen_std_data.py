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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/std_data'
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
import logging.config
import time
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data_standalone')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'exp_standalone_unmixing_CLSUnSAL_DC1')
CONFIG_FILE = os.path.join(DATA_DIR, 'standalone_unmixing_CLSUnSAL_DC1.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(CONFIG_FILE, 'r') as f:
    CONFIG = json.load(f)
EPS = CONFIG['EPS']
logging_config = {'version': 1, 'formatters': {'simple': {'format': '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', 'datefmt': '%d-%b-%y %H:%M:%S'}}, 'handlers': {'console': {'class': 'logging.StreamHandler', 'level': 'DEBUG', 'formatter': 'simple', 'stream': 'ext://sys.stdout'}}, 'root': {'level': 'DEBUG', 'handlers': ['console']}}
logging.config.dictConfig(logging_config)
log = logging.getLogger(__name__)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def set_seeds(seed):
    np.random.seed(seed)
set_seeds(CONFIG['seed'])
INTEGER_VALUES = ('H', 'W', 'M', 'L', 'p', 'N')

class HSI:

    def __init__(self, dataset: str, data_dir: str, figs_dir: str) -> None:
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
        assert os.path.isfile(path), f'Data file not found: {path}'
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def SVD_projection(Y, p):
    log.debug(f'Y shape => {Y.shape}')
    (V, SS, U) = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f'projected Y shape => {denoised_image_reshape.shape}')
    return np.clip(denoised_image_reshape, 0, 1)

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

class aRMSE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        (A, Aref) = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())

class SRE(BaseMetric):

    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        (X, Xref) = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, 'fro') / LA.norm(Xref - X, 'fro'))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/meta_data.json')
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
try:
    from munkres import Munkres
    HAS_MUNKRES = True
except ImportError:
    HAS_MUNKRES = False
    log.warning('munkres package not available. AbundancesAligner will not work.')

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
        if not HAS_MUNKRES:
            raise ImportError('munkres package required for HungarianAligner')
        self.dists = self.criterion(A.T, self.Aref.T)
        p = A.shape[0]
        P = np.zeros((p, p))
        m = Munkres()
        indices = m.compute(self.dists.tolist())
        for (row, col) in indices:
            P[row, col] = 1.0
        self.P = P.T

class AbundancesAligner(HungarianAligner):

    def __init__(self, **kwargs):
        super().__init__(criterion=MSE(), **kwargs)

class UnmixingModel:

    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f'{self.__class__.__name__}'
        return msg

    def print_time(self):
        return f'{self} took {self.time:.2f}s'

class SemiSupervisedUnmixingModel(UnmixingModel):

    def __init__(self):
        super().__init__()

    def compute_abundances(self, Y, D, *args, **kwargs):
        raise NotImplementedError(f'Solver is not implemented for {self}')

class CLSUnSAL(SemiSupervisedUnmixingModel):
    """
    CLSUnSAL - Collaborative Sparse Unmixing via Split Augmented Lagrangian
    
    Source: https://github.com/etienne-monier/lib-unmixing/blob/master/unmixing.py
    """

    def __init__(self, AL_iters=1000, lambd=0.0, verbose=True, tol=0.0001, mu=0.1, x0=0, *args, **kwargs):
        super().__init__()
        self.AL_iters = AL_iters
        self.lambd = lambd
        self.verbose = verbose
        self.tol = tol
        self.x0 = x0
        self.mu = mu

    def compute_abundances(self, Y, D, *args, **kwargs):
        tic = time.time()
        (LD, M) = D.shape
        (L, N) = Y.shape
        assert L == LD, 'Inconsistent number of channels for D and Y'
        lambd = self.lambd
        norm_d = np.sqrt(np.mean(D ** 2))
        log.debug(f'Norm D => {norm_d:.3e}')
        D = D / norm_d
        Y = Y / norm_d
        lambd = lambd / norm_d ** 2
        log.debug(f'Lambda initial value => {lambd:.3e}')
        mu = self.mu
        log.debug(f'Mu initial value => {mu:.3e}')
        (UF, sF, VF) = LA.svd(D.T @ D)
        IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
        AA = LA.inv(D.T @ D + 2 * np.eye(M))
        if self.x0 == 0:
            x = IF @ D.T @ Y
        else:
            x = self.x0
        u = x
        v1 = D @ x
        v2 = x
        v3 = x
        d1 = v1
        d2 = v2
        d3 = v3
        tol = np.sqrt(N * M) * self.tol
        log.debug(f'Tolerance => {tol:.3e}')
        k = 1
        res_p = float('inf')
        res_d = float('inf')
        while k <= self.AL_iters and (np.abs(res_p) > tol or np.abs(res_d) > tol):
            if k % 10 == 1:
                u0 = u
            u = AA @ (D.T @ (v1 + d1) + v2 + d2 + v3 + d3)
            v1 = (Y + mu * (D @ u - d1)) / (1 + mu)

            def current_fn(b):
                return self.vect_soft_thresh(b, lambd / mu)
            v2 = np.apply_along_axis(current_fn, axis=1, arr=u - d2)
            v3 = np.maximum(u - d3, 0)
            d1 = d1 - D @ u + v1
            d2 = d2 - u + v2
            d3 = d3 - u + v3
            if k % 10 == 1:
                res_p = LA.norm(D @ u - v1) + LA.norm(u - v2) + LA.norm(u - v3)
                res_d = mu * LA.norm(u - u0)
                if self.verbose:
                    log.info(f'k = {k}, res_p = {res_p:.3e}, res_d = {res_d:.3e}, mu = {mu:.3e}')
                if res_p > 10 * res_d:
                    mu = mu * 2
                if res_d > 10 * res_p:
                    mu = mu / 2
            k += 1
        Ahat = v3
        self.time = time.time() - tic
        log.info(self.print_time())
        return Ahat

    @staticmethod
    def vect_soft_thresh(b, t):
        max_b = np.maximum(LA.norm(b) - t, 0)
        ret = b * max_b / (max_b + t)
        return ret

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_CLSUnSAL_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    log.info('Semi-Supervised Unmixing - [START]...')
    cfg = CONFIG
    noise = AdditiveWhiteGaussianNoise(SNR=cfg['SNR'])
    hsi = HSIWithGT(dataset=cfg['dataset'], data_dir=DATA_DIR, figs_dir=OUTPUT_DIR)
    log.info(hsi)
    (Y, p, D) = hsi.get_data()
    (H, W) = hsi.get_img_shape()
    Y = noise.apply(Y)
    if cfg['l2_normalization']:
        Y = Y / np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
    if cfg['projection']:
        Y = SVD_projection(Y, p)
    model_cfg = cfg['model']
    model = CLSUnSAL(AL_iters=model_cfg['AL_iters'], lambd=model_cfg['lambd'], verbose=model_cfg['verbose'], tol=model_cfg['tol'], mu=model_cfg['mu'], x0=model_cfg['x0'])
    A_hat = model.compute_abundances(Y, D, p=p, H=H, W=W)
    E_hat = np.zeros((Y.shape[0], p))
    estimates_data = {'E': E_hat, 'A': A_hat.reshape(-1, H, W)}
    estimates_path = os.path.join(OUTPUT_DIR, 'estimates.mat')
    sio.savemat(estimates_path, estimates_data)
    log.info(f'Saved estimates to {estimates_path}')
    all_metrics = {}
    if hsi.has_GT():
        (_, A_gt) = hsi.get_GT()
        if cfg['force_align']:
            aligner = AbundancesAligner(Aref=A_gt)
            A1 = aligner.fit_transform(A_hat)
        else:
            index = hsi.get_index()
            A1 = A_hat[index]
        labels = hsi.get_labels()
        sre_metrics = compute_metric(SRE(), A_gt, A1, labels, detail=False, on_endmembers=False)
        all_metrics['SRE'] = sre_metrics
        armse_metrics = compute_metric(aRMSE(), A_gt, A1, labels, detail=True, on_endmembers=False)
        all_metrics['aRMSE'] = armse_metrics
        metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f'Saved metrics to {metrics_path}')
        hsi.plot_abundances(A0=None, run=0)
        gt_plot_src = os.path.join(OUTPUT_DIR, f"{cfg['dataset']}-abundances-GT.png")
        gt_plot_dst = os.path.join(OUTPUT_DIR, f"{cfg['dataset']}-abundances-GT.png")
        if os.path.exists(gt_plot_src) and gt_plot_src != gt_plot_dst:
            os.rename(gt_plot_src, gt_plot_dst)
        hsi.plot_abundances(A0=A1, run='estimated')
    log.info('Semi-Supervised Unmixing - [END]')
    log.info('=' * 60)
    log.info('RESULTS SUMMARY')
    log.info('=' * 60)
    log.info(f"Dataset: {cfg['dataset']}")
    log.info(f'Model: CLSUnSAL')
    log.info(f"SNR: {cfg['SNR']} dB")
    if 'SRE' in all_metrics:
        log.info(f"SRE: {all_metrics['SRE']['Overall']:.4f} dB")
    if 'aRMSE' in all_metrics:
        log.info(f"aRMSE: {all_metrics['aRMSE']['Overall']:.4f} %")
    log.info('=' * 60)
if __name__ == '__main__':
    main()