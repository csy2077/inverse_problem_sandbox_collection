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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/std_data'
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
import time
import logging
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spams
from munkres import Munkres
from tqdm import tqdm
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def rel_path(*args):
    """Convert relative path to absolute path based on script location."""
    return os.path.join(SCRIPT_DIR, *args)
CONFIG_PATH = rel_path('data_standalone', 'standalone_unmixing_SUnAA_DC1.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
EPS = CONFIG['EPS']
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
        path = rel_path(data_dir, filename)
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
        self.figs_dir = rel_path(figs_dir)
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

    def plot_endmembers(self, E0=None, suffix='-GT'):
        """
        Display endmembers
        """
        title = f'{self.name} - endmembers' + (' (GT)' if E0 is None else ' (Estimated)')
        ylabel = 'Reflectance'
        xlabel = '# Bands'
        E = np.copy(self.E) if E0 is None else np.copy(E0)
        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            label = self.labels[pp] if pp < len(self.labels) else f'#{pp}'
            plt.plot(E[:, pp], label=label)
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        figname = f'{self.name}-endmembers{suffix}.png'
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()
        log.info(f'Saved endmembers plot to {os.path.join(self.figs_dir, figname)}')

    def plot_abundances(self, A0=None, suffix='-GT'):
        (nrows, ncols) = (1, self.p)
        title = f'{self.name} - abundances' + (' (GT)' if A0 is None else ' (Estimated)')
        A = np.copy(self.A) if A0 is None else np.copy(A0)
        A = A.reshape(self.p, self.H, self.W)
        (fig, ax) = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj] if ncols > 1 else ax
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(A[kk], vmin=0.0, vmax=1.0)
                label = self.labels[kk] if kk < len(self.labels) else f'#{kk}'
                curr_ax.set_title(f'{label}')
                curr_ax.axis('off')
                fig.colorbar(mappable, ax=curr_ax, location='right', shrink=0.5)
                kk += 1
                if kk == self.p:
                    break
        plt.suptitle(title)
        figname = f'{self.name}-abundances{suffix}.png'
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()
        log.info(f'Saved abundances plot to {os.path.join(self.figs_dir, figname)}')

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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def SVD_projection(Y, p):
    log.debug(f'Y shape => {Y.shape}')
    (V, SS, U) = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f'projected Y shape => {denoised_image_reshape.shape}')
    return np.clip(denoised_image_reshape, 0, 1)

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

class SUnAA(SemiSupervisedUnmixingModel):

    def __init__(self, T, low_rank=False, *args, **kwargs):
        super().__init__()
        self.T = T
        self.low_rank = low_rank

    def compute_abundances(self, Y, D, p, *args, **kwargs):
        self.p = p

        def loss(a, b):
            return 0.5 * ((Y - D @ b @ a) ** 2).sum()

        def update_B(a, b):
            R = Y - D @ b @ a
            for jj in range(self.p):
                z_j = D @ b[:, jj]
                norm_aj = np.linalg.norm(a[jj])
                if norm_aj < 1e-10:
                    ZZ = z_j
                else:
                    ZZ = R @ a[jj] / norm_aj ** 2 + z_j
                bb = spams.decompSimplex(np.asfortranarray(ZZ[:, np.newaxis]), DD)
                b[:, jj] = np.squeeze(bb.todense())
                R = R + (z_j - D @ b[:, jj])[:, np.newaxis] @ a[jj][np.newaxis, :]
            return b
        tic = time.time()
        (_, N) = Y.shape
        (_, N_atoms) = D.shape
        YY = np.asfortranarray(Y)
        DD = np.asfortranarray(D)
        B = 1 / N_atoms * np.ones((N_atoms, self.p))
        A = 1 / self.p * np.ones((self.p, N))
        log.info(f'Initial loss => {loss(A, B):.2f}')
        progress = tqdm(range(self.T))
        for pp in progress:
            B = update_B(A, B)
            A = np.array(spams.decompSimplex(YY, np.asfortranarray(D @ B)).todense())
            progress.set_postfix_str(f'loss={loss(A, B):.2f}')
        tac = time.time()
        self.time = round(tac - tic, 2)
        log.info(f'{self} took {self.time}s')
        log.info(f'Final loss => {loss(A, B):.2f}')
        self.E_hat = D @ B
        self.B = B
        self.A_lowrank = A
        if self.low_rank:
            return A
        return B @ A

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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """
    Return individual and global metric
    """
    d = {}
    d['Overall'] = round(float(metric(X_hat, X_gt)), 4)
    if detail:
        for (ii, label) in enumerate(labels):
            if on_endmembers:
                (x_gt, x_hat) = (X_gt[:, ii][:, None], X_hat[:, ii][:, None])
                d[label] = round(float(metric(x_hat, x_gt)), 4)
            else:
                d[label] = round(float(metric(X_hat[ii], X_gt[ii])), 4)
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
        indices = m.compute(self.dists.tolist())
        for (row, col) in indices:
            P[row, col] = 1.0
        self.P = P.T

class AbundancesAligner(HungarianAligner):

    def __init__(self, **kwargs):
        super().__init__(criterion=MSE(), **kwargs)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def set_seeds(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_SUnAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    log.info('=' * 60)
    log.info('Standalone Semi-Supervised Unmixing - SUnAA on DC1')
    log.info('=' * 60)
    seed = CONFIG['seed']
    set_seeds(seed)
    log.info(f'Random seed set to: {seed}')
    noise = AdditiveWhiteGaussianNoise(SNR=CONFIG['SNR'])
    hsi = HSIWithGT(dataset=CONFIG['dataset'], data_dir=CONFIG['data_dir'], figs_dir=CONFIG['figs_dir'])
    log.info(hsi)
    (Y, p, D) = hsi.get_data()
    (H, W) = hsi.get_img_shape()
    Y = noise.apply(Y)
    if CONFIG['l2_normalization']:
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    if CONFIG['projection']:
        Y = SVD_projection(Y, p)
    model_config = CONFIG['model']
    model = SUnAA(T=model_config['T'], low_rank=model_config['low_rank'])
    log.info('Semi-Supervised Unmixing - [START]')
    A_hat = model.compute_abundances(Y, D, p=p, H=H, W=W)
    E_hat = np.zeros((Y.shape[0], p))
    estimates_path = rel_path(CONFIG['figs_dir'], 'estimates.mat')
    estimates_data = {'E': E_hat, 'A': A_hat.reshape(-1, H, W)}
    sio.savemat(estimates_path, estimates_data)
    log.info(f'Saved estimates to {estimates_path}')
    all_metrics = {}
    if hsi.has_GT():
        (_, A_gt) = hsi.get_GT()
        if CONFIG['force_align']:
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
        metrics_path = rel_path(CONFIG['figs_dir'], 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f'Saved metrics to {metrics_path}')
    log.info('Semi-Supervised Unmixing - [END]')
    log.info('=' * 60)
    log.info('RESULTS SUMMARY')
    log.info('=' * 60)
    for (metric_name, metric_values) in all_metrics.items():
        log.info(f'{metric_name}: {metric_values}')
    log.info('=' * 60)
if __name__ == '__main__':
    main()