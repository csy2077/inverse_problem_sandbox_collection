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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/std_data'
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
import numpy as np
import h5py as h5
import torch
import random
import itertools
import functools as ftools
from tqdm.auto import tqdm
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
from logzero import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t
from numpy.fft import fftshift as fftshift_o, fftn as fftn_o
try:
    from pyfftw.interfaces.numpy_fft import fftshift
except:
    from numpy.fft import fftshift

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def TilePlot(images, layout, figsize=(10, 10), **kwargs):
    """
    Create a tiled plot of multiple images.
    
    Args:
        images: Tuple of 2D arrays to plot
        layout: (rows, cols) tuple
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        fig, im, ax: Figure, image objects, and axes
    """
    (rows, cols) = layout
    (fig, ax) = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        ax = np.array([ax])
    else:
        ax = ax.flatten()
    im = []
    for (i, img) in enumerate(images):
        if i < len(ax):
            if np.iscomplexobj(img):
                img = np.abs(img)
            im.append(ax[i].imshow(img))
    return (fig, im, ax)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def fft1d(arr, n):
    """1D FFT along axis n with proper shifting."""
    return fftshift_t(fftn_t(fftshift_t(arr, dim=[n]), dim=[n], norm='ortho'), dim=[n])

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def ifft1d(arr, n):
    """1D inverse FFT along axis n with proper shifting."""
    return fftshift_t(ifftn_t(fftshift_t(arr, dim=[n]), dim=[n], norm='ortho'), dim=[n])

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def getBufferSize(coef_of_exp, shp):
    """Calculate buffer size for resampling."""
    bufsize = np.array([np.round(0.5 * n * (1.0 / c - 1.0)).astype(int) for (n, c) in zip(shp, coef_of_exp)])
    return bufsize

class GridPlugin:
    """Plugin for 3D grid setup for Fourier-based operations."""

    def setUpGrid(self):
        """Set up the 3D grid for FFT-based shearing operations."""
        grid_np = np.mgrid[-self._domainSize[0] // 2.0:self._domainSize[0] // 2.0, -self._domainSize[1] // 2.0:self._domainSize[1] // 2.0, -self._domainSize[2] // 2.0:self._domainSize[2] // 2.0]
        self._gridprod = [torch.from_numpy(fftshift(2.0 * np.pi * grid_np[(n + 2) % 3] * grid_np[(n + 1) % 3] / self._domainSize[n])).cuda() for n in range(3)]
        return

class MountPlugin(GridPlugin):
    """Plugin for mounting objects on a simulated diffractometer."""

    def __init__(self, size, R, Br, s0):
        self._domainSize = tuple((size for n in range(3)))
        self.initializeDiffractometer(R, Br, s0)
        return

    def initializeDiffractometer(self, R, Br, s0):
        self._delTheta = np.pi / 4.0
        self._axis_dict = {'X': 0, 'Y': 1, 'Z': 2}
        self.createBuffer(Br, s0)
        self.setUpGrid()
        self.createMask()
        self.getRotationParameters(R, 'XYZ')
        self.prepareToShear(Br)
        return

    def createBuffer(self, Br, s0):
        self._diffbuff = getBufferSize(np.diag(Br) / s0, self._domainSize)
        self._padder = torch.nn.ConstantPad3d(list(itertools.chain(*[[n, n] if n > 0 else [0, 0] for n in self._diffbuff[::-1]])), 0)
        self._skin = tuple(((0, N) if db <= 0 else (db, N + db) for (db, N) in zip(self._diffbuff, self._domainSize)))
        return

    def createMask(self):
        mask = np.ones(self._padder(torch.zeros(self._domainSize)).shape)
        for num in self._diffbuff:
            if num < 0:
                mask[-num:num, :, :] = 0.0
            mask = np.transpose(mask, np.roll([0, 1, 2], -1))
        self._mask = torch.from_numpy(1.0 - mask).cuda()
        return

    def shearPrincipal(self, ax, shift):
        self._rho = torch.fft.ifftn(shift * torch.fft.fftn(self._rho, dim=[ax], norm='ortho'), dim=[ax], norm='ortho')
        return

    def rotatePrincipal(self, angle, ax=2):
        ax1 = (ax + 1) % 3
        ax2 = (ax + 2) % 3
        (shift1, shift2) = tuple((torch.exp(1j * ang * self._gridprod[ax]) for ang in [np.tan(angle / 2.0), -np.sin(angle)]))
        for (x, sh) in zip([ax1, ax2, ax1], [shift1, shift2, shift1]):
            self.shearPrincipal(x, sh)
        return

    def prepareToShear(self, Br):
        self._Bshear = Br.T / np.diag(Br).reshape(1, -1).repeat(3, axis=0)
        self._shearShift = []
        for n in range(3):
            (np1, np2) = ((n + 1) % 3, (n + 2) % 3)
            (alpha, beta) = (self._Bshear[np1, n], self._Bshear[np2, n])
            (ax1, ax2) = ((n - 1) % 3, (n + 1) % 3)
            shift_arg = alpha * self._gridprod[ax1] + beta * self._gridprod[ax2]
            shift = torch.exp(1j * shift_arg)
            self._shearShift.append(shift)
        return

    def shearResampleObject(self):
        for ax in range(3):
            self.shearPrincipal(ax, self._shearShift[ax])
        return

    def splitAngle(self, angle):
        anglist = [np.sign(angle) * self._delTheta] * int(np.absolute(angle // self._delTheta))
        if angle < 0.0:
            anglist[-1] += angle % self._delTheta
        else:
            anglist.append(angle % self._delTheta)
        return anglist

    def getRotationParameters(self, R, euler_convention='XYZ'):
        convention = [self._axis_dict[k] for k in list(euler_convention)]
        eulers = Rotation.from_matrix(R).as_euler(euler_convention)
        self._axes = []
        self._eulers = []
        for (ang, ax) in zip(eulers[::-1], convention[::-1]):
            anglist = self.splitAngle(ang)
            self._axes.extend([ax] * len(anglist))
            self._eulers.extend(anglist)
        return

    def rotateObject(self):
        for (ang, ax) in zip(self._eulers, self._axes):
            self.rotatePrincipal(ang, ax)
        return

    def bulkResampleObject(self):
        self._rhopad = self._padder(fftshift_t(self._rho))
        n = self._diffbuff[0]
        if n < 0:
            self._rhopad[-n:n, :, :] = ifft1d((self._mask * fft1d(self._rhopad, 0))[-n:n, :, :], 0)
        elif n > 0:
            self._rhopad[n:-n, :, :] = fft1d(self._rhopad[n:-n, :, :], 0)
            self._rhopad = ifft1d(self._rhopad, 0)
        n = self._diffbuff[1]
        if n < 0:
            self._rhopad[:, -n:n, :] = ifft1d((self._mask * fft1d(self._rhopad, 1))[:, -n:n, :], 1)
        elif n > 0:
            self._rhopad[:, n:-n, :] = fft1d(self._rhopad[:, n:-n, :], 1)
            self._rhopad = ifft1d(self._rhopad, 1)
        n = self._diffbuff[2]
        if n < 0:
            self._rhopad[:, :, -n:n] = ifft1d((self._mask * fft1d(self._rhopad, 2))[:, :, -n:n], 2)
        elif n > 0:
            self._rhopad[:, :, n:-n] = fft1d(self._rhopad[:, :, n:-n], 2)
            self._rhopad = ifft1d(self._rhopad, 2)
        self.removeBuffer()
        return

    def removeBuffer(self):
        self._rho = fftshift_t(self._rhopad[self._skin[0][0]:self._skin[0][1], self._skin[1][0]:self._skin[1][1], self._skin[2][0]:self._skin[2][1]])
        return

    def mountObject(self):
        self.rotateObject()
        self.bulkResampleObject()
        self.shearResampleObject()
        return

    def getMountedObject(self):
        return self._rho

    def refreshObject(self, img_t):
        self._rho = img_t
        return

class LatticePlugin:
    """Plugin for 3D Bravais lattice calculations."""

    def getLatticeBases(self, a, b, c, al, bt, gm):
        """
        Calculate lattice bases from lattice parameters.
        
        Args:
            a, b, c: lengths of lattice basis vectors (Angstrom)
            al, bt, gm: Angular separations of basis vectors (degrees)
        """
        (a, b, c) = tuple((lat / 10.0 for lat in [a, b, c]))
        (al, bt, gm) = tuple((np.pi / 180.0 * ang for ang in [al, bt, gm]))
        p = (np.cos(al) - np.cos(bt) * np.cos(gm)) / (np.sin(bt) * np.sin(gm))
        q = np.sqrt(1.0 - p ** 2)
        self.basis_real = np.array([[a, b * np.cos(gm), c * np.cos(bt)], [0.0, b * np.sin(gm), c * p * np.sin(bt)], [0.0, 0.0, c * q * np.sin(bt)]])
        if np.linalg.det(self.basis_real) < 0.0:
            self.basis_real[-1, -1] *= -1.0
        self.basis_real = torch.from_numpy(self.basis_real).cuda()
        self.basis_reciprocal = torch.linalg.inv(self.basis_real.T)
        self.metricTensor = self.basis_real.T @ self.basis_real
        self.planeStepTensor = torch.linalg.inv(self.metricTensor)
        self.d_jumps_unitcell = 1.0 / torch.sqrt(self.planeStepTensor.sum(axis=0))
        return

    def getPlaneSeparations(self):
        """Calculate plane separations from Miller indices."""
        idx_arr = torch.from_numpy(np.array([list(idx) for idx in self.miller_idx]).T).cuda()
        d = 1.0 / torch.sqrt((idx_arr * (self.planeStepTensor @ idx_arr)).sum(axis=0))
        return d

    def getRotatedCrystalBases(self):
        """Apply crystal rotation to bases."""
        R = torch.from_numpy(self.database['global'].attrs['Rcryst']).cuda()
        self.basis_real = R @ self.basis_real
        self.basis_reciprocal = R @ self.basis_reciprocal
        return

class ObjectPlugin:
    """Plugin for creating tight-bound object in array."""

    def createObjectBB_full(self, n_scan):
        """
        Creates phase object within bounding box from FULL set of optimizable variables.
        """
        self.scl = self.x[-len(self.bragg_condition):][n_scan]
        self.__amp__ = 0.5 * (1.0 + torch.tanh(self.x[:self.N] / self.activ_a))
        self.__u__ = self.x[self.N:4 * self.N].reshape(3, -1)
        phs = self.peaks[n_scan] @ self.__u__
        objBB = self.scl * self.__amp__ * torch.exp(2j * np.pi * phs)
        return objBB.reshape(self.cubeSize, self.cubeSize, self.cubeSize)

    def buildFullBB(self):
        arr1 = [val * np.ones(self.N) for val in [-10.0 * self.activ_a, 0.0, 0.0, 0.0]]
        arr1.append(np.zeros(len(self.bragg_condition)))
        self.x = torch.tensor(np.concatenate(tuple(arr1))).cuda()
        self.x[self.these_only] = self.x_new
        return

    def createObjectBB_part(self, n_scan):
        """
        Creates phase object within bounding box from RESTRICTED set of optimizable variables.
        """
        self.buildFullBB()
        objBB = self.createObjectBB_full(n_scan)
        return objBB

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_multireflection_bcdi_adam_nodisloc_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def TV3D(arr):
    """
    Calculates total variation regularization of input 3D array,
    as the sum of total variation along each axis.
    """
    return torch.mean(torch.ravel(torch.abs(arr[1:, :, :] - arr[:-1, :, :])) + torch.ravel(torch.abs(arr[:, 1:, :] - arr[:, :-1, :])) + torch.ravel(torch.abs(arr[:, :, 1:] - arr[:, :, :-1])))

class OptimizerPlugin:
    """Plugin for optimizer methods for the multi-reflection BCDI problem."""

    def initializeOptimizer(self, optim_var, learning_rate, lambda_tv, minibatch_plan, default_iterations=3000):
        """
        Initialize the optimizer.
        
        minibatch_plan format:
        {
            'minibatches': [list of <N>]
            'minibatch_size': [list of <N>]
            'iterations_per_minibatch': [list of <N>]
        }
        """
        self.lr = learning_rate
        try:
            dummy = len(self.error)
        except:
            self.error = []
        self.optimizer = torch.optim.Adam([optim_var], lr=self.lr)
        if isinstance(lambda_tv, type(None)):
            self._lfun_ = self.lossfn
        else:
            self._ltv_ = lambda_tv
            self._lfun_ = self.lossTV
        if isinstance(minibatch_plan, type(None)):
            minibatch_plan = {'minibatches': [1], 'minibatch_size': [len(self.bragg_condition)], 'iterations_per_minibatch': [default_iterations]}
        self.buildCustomPlan(minibatch_plan)
        return

    def buildCustomPlan(self, minibatch_plan):
        N = list(range(len(self.peaks)))
        scheme = [[[sz, it]] * num for (sz, it, num) in zip(minibatch_plan['minibatch_size'], minibatch_plan['iterations_per_minibatch'], minibatch_plan['minibatches'])]
        self.optimization_plan = [[[random.sample(N, sch[0]), sch[1]] for sch in epoch] for epoch in scheme]
        return

    def unitCellClamp(self, tol=1e-06):
        """Clamps distortions to crystallographic unit cell."""
        self.__u__ = self.x[self.N:-len(self.bragg_condition)].reshape(3, -1)
        temp = torch.linalg.solve(self.basis_real, self.__u__)
        with torch.no_grad():
            temp[:] = temp.clamp(-0.5 + tol, 0.5 - tol)
            self.x[self.N:-len(self.bragg_condition)].copy_((self.basis_real @ temp).ravel())
        return

    def lossfn(self):
        return sum(self.losses)

    def penaltyTV(self):
        mag = self.x[:self.N].reshape(*(self.cubeSize for p in range(3)))
        tvmag = TV3D(mag)
        return tvmag

    def lossTV(self):
        return self.lossfn() + self._ltv_ * self.penaltyTV()

    def run(self, epochs=3000):
        for epoch in tqdm(self.optimization_plan, desc='Epoch          ', total=len(self.optimization_plan)):
            for batch in tqdm(epoch, desc='Batch/minibatch', total=len(epoch), leave=False):
                for iteration in tqdm(range(batch[1]), desc='Iteration      ', leave=False):
                    self.optimizer.zero_grad()
                    self.losses = []
                    for n in batch[0]:
                        rho_m = self.getObjectInMount(n)
                        frho_m = fftn_t(rho_m, norm='ortho')
                        self.losses.append(torch.mean((torch.abs(frho_m) - self.bragg_condition[n]['data']) ** 2))
                    self.loss_total = self._lfun_()
                    self.loss_total.backward()
                    self.optimizer.step()
                    self.error.append(float(self.loss_total.cpu()))
                    self.unitCellClamp()
            self.medianFilter()
        return

class MultiReflectionSolver(LatticePlugin, ObjectPlugin, OptimizerPlugin):
    """
    Multi-reflection BCDI optimization solver.
    
    This class combines lattice, object, and optimizer plugins to perform
    multi-reflection Bragg coherent diffraction imaging reconstruction.
    """

    def __init__(self, database, signal_label='signal', size=128, sigma=3.0, activation_parameter=0.75, learning_rate=0.01, minibatch_plan=None, lambda_tv=None, medfilter_kernel=3, init_amplitude_state=None, scale_method='photon_max', cuda_device=0):
        self.loadGlobals(database)
        self.getLatticeBases(*tuple(self.globals['lattice_parms']))
        self.getRotatedCrystalBases()
        self.determineDimensions(size, sigma)
        self.createObjectBB = self.createObjectBB_full
        self.prepareScans(label=signal_label)
        self.createUnknowns(activation_parameter, init_amplitude_state, scale_method)
        self.initializeOptimizer(self.x, learning_rate=learning_rate, lambda_tv=lambda_tv, minibatch_plan=minibatch_plan)
        self.prepareMedianFilter(medfilter_kernel)
        return

    def prepareMedianFilter(self, kernel):
        self.mfk = kernel
        return

    def loadGlobals(self, dbase):
        self.database = dbase
        self.globals = dict(self.database['global'].attrs)
        return

    def determineDimensions(self, size, sigma):
        self.size = size
        self._domainSize = tuple((self.size for n in range(3)))
        self.cubeSize = np.round(size / sigma).astype(int)
        self.cubeSize += self.cubeSize % 2
        buff = (size - self.cubeSize) // 2
        self._buffer = torch.nn.ConstantPad3d(tuple((buff for n in range(6))), 0)
        return

    def createUnknowns(self, activation_parameter, init_amplitude_state, scale_method):
        self.activ_a = activation_parameter
        self.N = self.cubeSize ** 3
        if isinstance(init_amplitude_state, type(None)):
            mag = 2.0 * np.ones(self.N)
        else:
            mag = init_amplitude_state
        u = np.zeros(3 * self.N)
        norms = self.setScalingFactors(mag, scale_method)
        self.x = torch.from_numpy(np.concatenate((mag, u, norms))).cuda().requires_grad_()
        return

    def setScalingFactors(self, init, method):
        norms = []
        init_t = torch.from_numpy(init).cuda()
        for n in range(len(self.bragg_condition)):
            __amp__ = 0.5 * (1.0 + torch.tanh(init_t / self.activ_a))
            __u__ = torch.from_numpy(np.zeros((3, self.N))).cuda()
            phs = self.peaks[n] @ __u__
            objBB = (__amp__ * torch.exp(2j * np.pi * phs)).reshape(*(self.cubeSize for n in range(3)))
            self.bragg_condition[n]['mount'].refreshObject(fftshift_t(self._buffer(objBB)))
            self.bragg_condition[n]['mount'].mountObject()
            rho_m = self.bragg_condition[n]['mount'].getMountedObject()
            if method == 'photon_max':
                frho_m = fftshift_t(fftn_t(fftshift_t(rho_m), norm='ortho')).detach().cpu().numpy()
                scl = np.sqrt(self.bragg_condition[n]['data'].detach().cpu().numpy().max() / (np.absolute(frho_m) ** 2).max())
                norms.append(scl)
            elif method == 'energy':
                norms.append(self.bragg_condition[n]['data'].detach().cpu().numpy().sum() / (np.absolute(rho_m.detach().cpu().numpy()) ** 2).sum())
            else:
                logger.error("Should set either 'photon_max' or 'energy' for scale_method.")
                return []
        return np.sqrt(np.array(norms))

    def resetUnknowns(self, x):
        self.x = torch.from_numpy(x).cuda().requires_grad_()
        return

    def prepareScans(self, label):
        self.bragg_condition = []
        self.peaks = []
        self.miller_idx = []
        self.scan_list = ['scans/dataset_%d' % m for m in self.database['scans'].attrs['successful_scans']]
        for scan in self.scan_list:
            self.peaks.append(torch.from_numpy(self.database[scan].attrs['peak'][np.newaxis, :] * 10.0).cuda())
            self.miller_idx.append(self.database[scan].attrs['miller_idx'])
            scan_data = self.database['%s/%s' % (scan, label)][:]
            R = self.database[scan].attrs['RtoBragg']
            Bdet = self.database[scan].attrs['Bdet']
            Br = self.database[scan].attrs['Breal']
            mnt = MountPlugin(self.size, Bdet.T @ R, Br, self.globals['step_cubic'])
            bcond = {'data': torch.from_numpy(fftshift(np.sqrt(scan_data))).cuda(), 'mount': mnt}
            self.bragg_condition.append(bcond)
        self.d_spacing = self.getPlaneSeparations()
        self.d_jumps = [mg ** 2 * pk.T for (mg, pk) in zip(self.d_spacing, self.peaks)]
        return

    def getObjectInMount(self, n):
        obj = self.createObjectBB(n)
        self.bragg_condition[n]['mount'].refreshObject(fftshift_t(self._buffer(obj)))
        self.bragg_condition[n]['mount'].mountObject()
        rho_m = self.bragg_condition[n]['mount'].getMountedObject()
        return rho_m

    def centerObject(self):
        state = self.x[:self.N].reshape(*(self.cubeSize for n in range(3)))
        amp = 0.5 * (1.0 + torch.tanh(state / self.activ_a)).detach().cpu().numpy()
        u = self.x[self.N:4 * self.N].reshape(3, -1).detach().cpu().numpy()
        (ux, uy, uz) = tuple((arr.reshape(*(self.cubeSize for n in range(3))) for arr in u))
        grid = np.mgrid[-self.cubeSize // 2:self.cubeSize // 2, -self.cubeSize // 2:self.cubeSize // 2, -self.cubeSize // 2:self.cubeSize // 2]
        shift = [-np.round((arr * amp).sum() / amp.sum()).astype(int) for arr in grid]
        state_c = np.roll(state.detach().cpu().numpy(), shift, axis=[0, 1, 2])
        (ux_c, uy_c, uz_c) = tuple((np.roll(arr, shift, axis=[0, 1, 2]) for arr in [ux, uy, uz]))
        xi = self.x[-len(self.bragg_condition):].detach().cpu().numpy()
        my_x = np.concatenate(tuple((arr.ravel() for arr in [state_c, ux_c, uy_c, uz_c, xi])))
        new_x = torch.from_numpy(my_x).requires_grad_().cuda()
        with torch.no_grad():
            self.x.copy_(new_x)
        return

    def setUpRestrictedOptimization(self, new_plan, amp_threshold=0.1, lambda_tv=None, learning_rate=None, median_filter=False):
        self.getSupportVars(amp_threshold, median_filter=median_filter)
        self.createObjectBB = self.createObjectBB_part
        if not isinstance(lambda_tv, type(None)):
            self._ltv_ = lambda_tv
        if not isinstance(learning_rate, type(None)):
            self.lr = learning_rate
        self.initializeOptimizer(self.x_new, learning_rate=self.lr, lambda_tv=self._ltv_, minibatch_plan=new_plan)
        return

    def getSupportVars(self, amp_threshold, median_filter):
        self.bin = []
        if median_filter:
            my_x = self.medianFilter()
        else:
            my_x = self.x.detach().cpu().numpy()
        ln = my_x.size
        self.bin.append(my_x)
        state = my_x[:self.N]
        amp = 0.5 * (1.0 + np.tanh(state / self.activ_a))
        here_c = np.where(amp > amp_threshold)[0]
        here_c = np.concatenate(tuple((n * self.N + here_c for n in range(4))))
        here_c = np.concatenate((here_c, np.array([ln - len(self.bragg_condition) + n for n in range(len(self.bragg_condition))])))
        self.bin.append(here_c)
        my_new_x = my_x[here_c]
        self.x_new = torch.from_numpy(my_new_x).cuda().requires_grad_()
        these = np.zeros(my_x.size)
        these[here_c] = 1.0
        self.these_only = torch.tensor(these, dtype=torch.bool).cuda()
        return

    def medianFilter(self):
        my_x = self.x.detach().cpu().numpy()
        arr = my_x[:-len(self.bragg_condition)]
        scalers = my_x[-len(self.bragg_condition):]
        arr_by4 = arr.reshape(4, -1)
        (state, ux, uy, uz) = tuple((ar.reshape(*(self.cubeSize for n in range(3))) for ar in arr_by4))
        amp = 0.5 * (1.0 + np.tanh(state / self.activ_a))
        (ux, uy, uz) = tuple((median_filter(amp * ar, size=self.mfk) for ar in [ux, uy, uz]))
        arr_out = np.concatenate(tuple((ar.ravel() for ar in [state, ux, uy, uz, scalers])))
        with torch.no_grad():
            self.x.copy_(torch.from_numpy(arr_out).cuda())
        return arr_out
if __name__ == '__main__':
    datafile = './data/Au_Nodisloc.h5'
    resultfile = './reconstructions/Results_nodisloc.h5'
    optim_plan = {'minibatches': [1, 1, 1, 1], 'minibatch_size': [2, 3, 4, 5], 'iterations_per_minibatch': [1, 1, 1, 20]}
    logger.info('Loading grain data into solver.')
    data = h5.File(datafile, 'r')
    logger.info('Printing data tree...')
    data.visit(print)
    data['scans'].attrs.keys()
    solver = MultiReflectionSolver(database=data, signal_label='signal/photon_max_100000.0', sigma=2.85, learning_rate=0.01, activation_parameter=1.0, minibatch_plan=optim_plan, lambda_tv=1e-05)
    logger.info('Created solver.')
    logger.info('Successful scans: %s' % str(data['scans'].attrs['successful_scans']))
    logger.info('Global parameters: ')
    for (key, value) in solver.globals.items():
        logger.info('\t%s: %s' % (key, value))
    logger.info('Running optimizer...')
    solver.run()
    solver.centerObject()
    amp = solver.__amp__.reshape(*(solver.cubeSize for n in range(3))).detach().cpu()
    u_recon = solver.__u__.detach().cpu()
    fig = plt.figure()
    plt.semilogy(solver.error)
    plt.xlabel('Iteration')
    plt.ylabel('$\\mathcal{L}(A, u)$')
    plt.grid()
    plt.title('Objective function (Poisson-stabilized)')
    fig.savefig('objective_function_phase1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved objective_function_phase1.png')
    (fig, im, ax) = TilePlot(tuple((np.transpose(amp, np.roll([0, 1, 2], n))[:, :, 23] for n in range(3))), (1, 3), (15, 5))
    fig.suptitle('Reconstructed electron density (normalized)')
    for (n, st) in enumerate(['XY', 'YZ', 'ZX']):
        ax[n].set_title(st)
    fig.savefig('electron_density_phase1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved electron_density_phase1.png')
    (fig, im, ax) = TilePlot(tuple((np.transpose(arr.reshape(46, 46, 46), np.roll([0, 1, 2], n))[:, :, 23] for n in range(3) for arr in u_recon)), (3, 3), (15, 12))
    fig.savefig('displacement_field_phase1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved displacement_field_phase1.png')
    new_plan = {'minibatches': [1], 'minibatch_size': [5], 'iterations_per_minibatch': [20]}
    solver.setUpRestrictedOptimization(new_plan=new_plan, amp_threshold=0.2, learning_rate=0.005)
    logger.info('Running optimizer...')
    solver.run()
    fig = plt.figure()
    plt.semilogy(solver.error)
    plt.xlabel('Iteration')
    plt.ylabel('$\\mathcal{L}(A, u)$')
    plt.grid()
    plt.title('Objective function (Poisson-stabilized)')
    fig.savefig('objective_function_phase2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved objective_function_phase2.png')
    (fig, im, ax) = TilePlot(tuple((np.transpose(amp, np.roll([0, 1, 2], n))[:, :, 23] for n in range(3))), (1, 3), (15, 5))
    fig.suptitle('Reconstructed electron density (normalized)')
    for (n, st) in enumerate(['XY', 'YZ', 'ZX']):
        ax[n].set_title(st)
    fig.savefig('electron_density_phase2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved electron_density_phase2.png')
    (fig, im, ax) = TilePlot(tuple((np.transpose(arr.reshape(46, 46, 46), np.roll([0, 1, 2], n))[:, :, 23] for n in range(3) for arr in u_recon)), (3, 3), (15, 12))
    fig.savefig('displacement_field_phase2.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved displacement_field_phase2.png')
    data.close()
    res = h5.File(resultfile, 'w')
    res.create_group('real_space')
    res['real_space'].create_dataset('electron_density', data=amp)
    res['real_space'].create_dataset('lattice_deformation', data=u_recon)
    res['real_space'].attrs['scale'] = list(solver.x[-len(solver.bragg_condition):].detach().cpu().numpy())
    res.create_dataset('reconstruction_error', data=solver.error)
    res.create_group('reciprocal_space')
    res['reciprocal_space'].attrs['scan_names'] = solver.scan_list
    data = h5.File(datafile, 'r')
    for (n, scan) in enumerate([st.replace('scans/', '') for st in solver.scan_list]):
        print(n, scan)
        obj = solver.getObjectInMount(n)
        fobj = fftshift_t(fftn_t(obj)).detach().cpu().numpy()
        data_arr = fftshift_o(solver.bragg_condition[n]['data'].detach().cpu().numpy()) ** 2
        res['reciprocal_space'].create_group(scan)
        res['reciprocal_space/%s' % scan].create_dataset('wave', data=fobj)
        res['reciprocal_space/%s' % scan].create_dataset('data', data=data_arr)
    data.close()
    res.close()
    logger.info('Results saved to %s' % resultfile)
    logger.info('Done!')