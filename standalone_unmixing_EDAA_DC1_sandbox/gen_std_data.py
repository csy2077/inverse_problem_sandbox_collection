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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_EDAA_DC1_sandbox/run_code/std_data'
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
from pathlib import Path
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from munkres import Munkres
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data_standalone'
OUTPUT_DIR = SCRIPT_DIR / 'exp_standalone'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = DATA_DIR / 'standalone_unmixing_EDAA_DC1.json'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
EPS = CONFIG.get('EPS', 1e-10)

class AdditiveWhiteGaussianNoise:

    def __init__(self, SNR=None):
        self.SNR = SNR

    def apply(self, Y):
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_EDAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def SVD_projection(Y, p):
    log.debug(f'Y shape => {Y.shape}')
    (V, SS, U) = np.linalg.svd(Y, full_matrices=False)
    PC = np.diag(SS) @ U
    denoised_image_reshape = V[:, :p] @ PC[:p]
    log.debug(f'projected Y shape => {denoised_image_reshape.shape}')
    return np.clip(denoised_image_reshape, 0, 1)
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

    def get_img_shape(self):
        return (self.H, self.W)

    def get_labels(self):
        return self.labels

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
        title = f'{self.name} - endmembers' + (' (GT)' if E0 is None else '')
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
        suffix = '-GT' if E0 is None else f'-{run}'
        figname = f'{self.name}-endmembers{suffix}.png'
        plt.savefig(os.path.join(self.figs_dir, figname))
        plt.close()

    def plot_abundances(self, A0=None, run=0):
        (nrows, ncols) = (1, self.p)
        title = f'{self.name} - abundances' + (' (GT)' if A0 is None else '')
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
                label = self.labels[kk] if kk < len(self.labels) else f'#{kk}'
                curr_ax.set_title(f'{label}')
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

class UnmixingModel:

    def __init__(self):
        self.time = 0

    def __repr__(self):
        return f'{self.__class__.__name__}'

    def print_time(self):
        return f'{self} took {self.time:.2f}s'

class BlindUnmixingModel(UnmixingModel):

    def __init__(self):
        super().__init__()

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):
        raise NotImplementedError(f'Solver is not implemented for {self}')

class EDAA(BlindUnmixingModel):

    def __init__(self, T=100, K1=5, K2=5, M=50, normalize=True, *args, **kwargs):
        super().__init__()
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.M = M
        self.normalize = normalize
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        best_E = None
        best_A = None
        (_, N) = Y.shape

        def loss(a, b):
            return 0.5 * ((Y - Y @ b @ a) ** 2).sum()

        def residual_l1(a, b):
            return (Y - Y @ b @ a).abs().sum()

        def grad_A(a, b):
            YB = Y @ b
            ret = -YB.t() @ (Y - YB @ a)
            return ret

        def grad_B(a, b):
            return -Y.t() @ ((Y - Y @ b @ a) @ a.t())

        def update(a, b):
            return F.softmax(torch.log(a) + b, dim=0)

        def computeLA(b):
            YB = Y @ b
            S = torch.linalg.svdvals(YB)
            return S[0] * S[0]

        def max_correl(e):
            return np.max(np.corrcoef(e.T) - np.eye(p))
        results = {}
        tic = time.time()
        Y = torch.Tensor(Y)
        for m in tqdm(range(self.M)):
            torch.manual_seed(m + seed)
            generator = np.random.RandomState(m + seed)
            with torch.no_grad():
                B = F.softmax(0.1 * torch.rand((N, p)), dim=0)
                A = 1 / p * torch.ones((p, N))
                Y = Y.to(self.device)
                A = A.to(self.device)
                B = B.to(self.device)
                factA = 2 ** generator.randint(-3, 4)
                self.etaA = factA / computeLA(B)
                self.etaB = self.etaA * (p / N) ** 0.5
                for _ in range(self.T):
                    for _ in range(self.K1):
                        A = update(A, -self.etaA * grad_A(A, B))
                    for _ in range(self.K2):
                        B = update(B, -self.etaB * grad_B(A, B))
                fit_m = residual_l1(A, B).item()
                E = (Y @ B).cpu().numpy()
                A = A.cpu().numpy()
                B = B.cpu().numpy()
                Rm = max_correl(E)
                results[m] = {'Rm': Rm, 'Em': E, 'Am': A, 'Bm': B, 'fit_m': fit_m, 'factA': factA}
        min_fit_l1 = np.min([v['fit_m'] for v in results.values()])

        def fit_l1_cutoff(idx, tol=0.05):
            val = results[idx]['fit_m']
            return abs(val - min_fit_l1) / abs(val) < tol
        sorted_indices = sorted(filter(fit_l1_cutoff, results), key=lambda x: results[x]['Rm'])
        best_result_idx = sorted_indices[0]
        best_result = results[best_result_idx]
        best_E = best_result['Em']
        best_A = best_result['Am']
        self.B = best_result['Bm']
        toc = time.time()
        self.time = toc - tic
        log.info(self.print_time())
        return (best_E, best_A)

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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_EDAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_unmixing_EDAA_DC1_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    log.info('Blind Unmixing - [START]')
    seed = CONFIG.get('seed', 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    SNR = CONFIG.get('SNR', None)
    noise = AdditiveWhiteGaussianNoise(SNR=SNR)
    hsi = HSIWithGT(dataset=CONFIG['data']['dataset'], data_dir=str(DATA_DIR), figs_dir=str(OUTPUT_DIR))
    log.info(hsi)
    (Y, p, _) = hsi.get_data()
    (H, W) = hsi.get_img_shape()
    Y = noise.apply(Y)
    if CONFIG.get('l2_normalization', False):
        normY = np.linalg.norm(Y, axis=0, ord=2, keepdims=True)
        Y = Y / normY
    if CONFIG.get('projection', False):
        Y = SVD_projection(Y, p)
    edaa_cfg = CONFIG['EDAA']
    model = EDAA(T=edaa_cfg['T'], K1=edaa_cfg['K1'], K2=edaa_cfg['K2'], M=edaa_cfg['M'], normalize=edaa_cfg.get('normalize', True))
    (E_hat, A_hat) = model.compute_endmembers_and_abundances(Y, p, H=H, W=W)
    estimates_path = OUTPUT_DIR / 'estimates.mat'
    sio.savemat(str(estimates_path), {'E': E_hat, 'A': A_hat.reshape(-1, H, W)})
    log.info(f'Estimates saved to {estimates_path}')
    all_metrics = {}
    if hsi.has_GT():
        (E_gt, A_gt) = hsi.get_GT()
        aligner = AbundancesAligner(Aref=A_gt)
        A1 = aligner.fit_transform(A_hat)
        E1 = aligner.transform_endmembers(E_hat)
        labels = hsi.get_labels()
        all_metrics['SRE'] = compute_metric(SRE(), A_gt, A1, labels, detail=False, on_endmembers=False)
        all_metrics['aRMSE'] = compute_metric(aRMSE(), A_gt, A1, labels, detail=True, on_endmembers=False)
        all_metrics['SAD'] = compute_metric(SADDegrees(), E_gt, E1, labels, detail=True, on_endmembers=True)
        all_metrics['eRMSE'] = compute_metric(eRMSE(), E_gt, E1, labels, detail=True, on_endmembers=True)
        metrics_path = OUTPUT_DIR / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f'Metrics saved to {metrics_path}')
        hsi.E = E1
        hsi.plot_endmembers(E0=E1, run='estimated')
        hsi.plot_abundances(A0=A1, run='estimated')
    log.info('Blind Unmixing - [END]')
if __name__ == '__main__':
    main()