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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/std_data'
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
import sys
import numpy as np
import tifffile
from read_roi import read_roi_file
from functools import reduce
import operator
import astra
from astra import data2d, projector, creators, algorithm, functions
import scipy.sparse.linalg

class OpTomo(scipy.sparse.linalg.LinearOperator):
    """Object that imitates a projection matrix with a given projector."""

    def __init__(self, proj_id):
        self.dtype = np.float32
        try:
            self.vg = projector.volume_geometry(proj_id)
            self.pg = projector.projection_geometry(proj_id)
            self.data_mod = data2d
            self.appendString = ''
            if projector.is_cuda(proj_id):
                self.appendString += '_CUDA'
        except Exception:
            from astra import data3d, projector3d
            self.vg = projector3d.volume_geometry(proj_id)
            self.pg = projector3d.projection_geometry(proj_id)
            self.data_mod = data3d
            self.appendString = '3D'
            if projector3d.is_cuda(proj_id):
                self.appendString += '_CUDA'
        self.vshape = functions.geom_size(self.vg)
        self.vsize = reduce(operator.mul, self.vshape)
        self.sshape = functions.geom_size(self.pg)
        self.ssize = reduce(operator.mul, self.sshape)
        self.shape = (self.ssize, self.vsize)
        self.proj_id = proj_id
        self.transposeOpTomo = OpTomoTranspose(self)
        try:
            self.T = self.transposeOpTomo
        except AttributeError:
            pass

    def _transpose(self):
        return self.transposeOpTomo

    def __checkArray(self, arr, shp):
        if len(arr.shape) == 1:
            arr = arr.reshape(shp)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.flags['C_CONTIGUOUS'] == False:
            arr = np.ascontiguousarray(arr)
        return arr

    def _matvec(self, v):
        return self.FP(v, out=None).ravel()

    def rmatvec(self, s):
        return self.BP(s, out=None).ravel()

    def __mul__(self, v):
        if isinstance(v, np.ndarray) and v.shape == self.vshape:
            return self._matvec(v)
        return scipy.sparse.linalg.LinearOperator.__mul__(self, v)

    @_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
    @_data_capture_decorator_
    def reconstruct(self, method, s, iterations=1, extraOptions=None):
        """Reconstruct an object using the specified method."""
        if extraOptions == {}:
            opts = {}
        if extraOptions is None:
            opts = {}
        s = self.__checkArray(s, self.sshape)
        sid = self.data_mod.link('-sino', self.pg, s)
        v = np.zeros(self.vshape, dtype=np.float32)
        vid = self.data_mod.link('-vol', self.vg, v)
        cfg = creators.astra_dict(method)
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        if 'FilterType' in list(extraOptions.keys()):
            cfg['FilterType'] = extraOptions['FilterType']
            opts = {key: extraOptions[key] for key in extraOptions if key != 'FilterType'}
        else:
            opts = extraOptions
        cfg['option'] = opts
        alg_id = algorithm.create(cfg)
        algorithm.run(alg_id, iterations)
        algorithm.delete(alg_id)
        self.data_mod.delete([vid, sid])
        return v

    def FP(self, v, out=None):
        """Perform forward projection."""
        v = self.__checkArray(v, self.vshape)
        vid = self.data_mod.link('-vol', self.vg, v)
        if out is None:
            out = np.zeros(self.sshape, dtype=np.float32)
        sid = self.data_mod.link('-sino', self.pg, out)
        cfg = creators.astra_dict('FP' + self.appendString)
        cfg['ProjectionDataId'] = sid
        cfg['VolumeDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        fp_id = algorithm.create(cfg)
        algorithm.run(fp_id)
        algorithm.delete(fp_id)
        self.data_mod.delete([vid, sid])
        return out

    def BP(self, s, out=None):
        """Perform backprojection."""
        s = self.__checkArray(s, self.sshape)
        sid = self.data_mod.link('-sino', self.pg, s)
        if out is None:
            out = np.zeros(self.vshape, dtype=np.float32)
        vid = self.data_mod.link('-vol', self.vg, out)
        cfg = creators.astra_dict('BP' + self.appendString)
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        cfg['ProjectorId'] = self.proj_id
        bp_id = algorithm.create(cfg)
        algorithm.run(bp_id)
        algorithm.delete(bp_id)
        self.data_mod.delete([vid, sid])
        return out

class OpTomoTranspose(scipy.sparse.linalg.LinearOperator):
    """Transpose operation of the OpTomo object."""

    def __init__(self, parent):
        self.parent = parent
        self.dtype = np.float32
        self.shape = (parent.shape[1], parent.shape[0])
        try:
            self.T = self.parent
        except AttributeError:
            pass

    def _matvec(self, s):
        return self.parent.rmatvec(s)

    def rmatvec(self, v):
        return self.parent.matvec(v)

    def _transpose(self):
        return self.parent

    def __mul__(self, s):
        if isinstance(s, np.ndarray) and s.shape == self.parent.sshape:
            return self._matvec(s)
        return scipy.sparse.linalg.LinearOperator.__mul__(self, s)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def read_tiff(fname):
    """Read a 2D TIFF image."""
    fname = os.path.abspath(fname)
    img = tifffile.imread(fname)
    return img

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def write_tiff(fname, img, overwrite=True):
    """Write an array to a TIFF file."""
    if not (fname.endswith('.tif') or fname.endswith('.tiff')):
        fname = fname + '.tiff'
    tifffile.imsave(fname, img)
    print(f'File saved: {fname}')

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def write_tiff_stack(fname, data, axis=0, start=0, digit=4, dtype=None, overwrite=True):
    """Write a 3D array to a stack of 2D TIFF images."""
    if data.ndim != 3:
        raise ValueError('Array must have 3 dimensions.')
    folder = os.path.dirname(fname)
    if folder == '':
        raise ValueError('File path not valid. Please specify the file path with folder.')
    if not os.path.isdir(folder):
        raise ValueError(f'Folder does not exist: {folder}')
    if dtype is None:
        dtype = data.dtype
    if axis != 0:
        data = np.swapaxes(data, 0, axis)
    nslice = data.shape[0]
    print(f"> Saving stack of images: {fname}_{'#' * digit}.tiff")
    for iz in range(nslice):
        outfile = fname + '_' + str(iz + start).zfill(digit) + '.tiff'
        tifffile.imsave(outfile, data[iz].astype(dtype))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def get_rect_coordinates_from_roi(fname):
    """Get rectangular coordinates from an ImageJ ROI file."""
    roi = read_roi_file(fname)
    l = list(roi.keys())
    height = roi[l[0]]['height']
    width = roi[l[0]]['width']
    top = roi[l[0]]['top']
    left = roi[l[0]]['left']
    rowmin = int(top)
    rowmax = int(top + height - 1)
    colmin = int(left)
    colmax = int(left + width - 1)
    return (rowmin, rowmax, colmin, colmax)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def get_astra_proj_matrix(nd, angles, method):
    """Get ASTRA projection matrix operator."""
    vol_geom = astra.create_vol_geom(nd, nd)
    proj_geom = astra.create_proj_geom('parallel', 1.0, nd, angles)
    if method.endswith('CUDA') or method.startswith('NN-FBP'):
        pid = astra.create_projector('cuda', proj_geom, vol_geom)
    else:
        pid = astra.create_projector('linear', proj_geom, vol_geom)
    pmat = OpTomo(pid)
    return pmat

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def recon_slice(sinogram, method, pmat, parameters=None, pixel_size=1.0, offset=0):
    """Reconstruct a single sinogram slice."""
    if type(offset) is float:
        offset = round(offset)
    if parameters is None:
        parameters = {}
    if 'iterations' in list(parameters.keys()):
        iterations = parameters['iterations']
        opts = {key: parameters[key] for key in parameters if key != 'iterations'}
    else:
        iterations = 1
        opts = parameters
    pixel_size = float(pixel_size)
    sinogram = sinogram / pixel_size
    if offset:
        sinogram = np.roll(sinogram, -offset, axis=1)
    rec = pmat.reconstruct(method, sinogram, iterations=iterations, extraOptions=opts)
    return rec

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def recon_stack(proj, method, pmat, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
    """Reconstruct a stack of sinograms or projections."""
    if parameters is None:
        parameters = {}
    if not sinogram_order:
        proj = np.swapaxes(proj, 0, 1)
    (nslice, na, nd) = proj.shape
    rec = np.zeros((nslice, nd, nd), dtype=np.float32)
    for s in range(nslice):
        print(f'\r  Reconstructing slice {s + 1}/{nslice}', end='')
        rec[s] = recon_slice(proj[s], method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset)
    print()
    return rec

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def reconstruct(tomo, angles, method, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
    """Reconstruct tomographic data using specified method."""
    if tomo.ndim != 2 and tomo.ndim != 3:
        raise ValueError('Invalid shape of array. Must have 2 or 3 dimensions.')
    nd = tomo.shape[-1]
    pmat = get_astra_proj_matrix(nd, angles, method)
    if tomo.ndim == 2:
        out = recon_slice(tomo, method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset)
    elif tomo.ndim == 3:
        out = recon_stack(tomo, method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset, sinogram_order=sinogram_order)
    else:
        raise ValueError('Invalid array dimensions. Must be 2D or 3D.')
    return out

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def get_circular_mask(nrow, ncol, radius=None, center=None):
    """Get a boolean circular mask."""
    if radius is None:
        radius = min(ncol, nrow) / 2
    if center is None:
        yc = ncol / 2.0
        xc = nrow / 2.0
    else:
        (yc, xc) = center
    ny = np.arange(ncol)
    nx = np.arange(nrow)
    (x, y) = np.meshgrid(nx, ny)
    mask = (y - yc + 0.5) ** 2 + (x - xc + 0.5) ** 2 < radius ** 2
    return mask

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def CNR(img, croi_signal=[], croi_background=[], froi_signal=[], froi_background=[]):
    """Compute Contrast-to-Noise Ratio."""
    if img.ndim != 2:
        raise ValueError('Input array must have 2 dimensions.')
    if croi_signal:
        (rowmin, rowmax, colmin, colmax) = croi_signal
    if froi_signal:
        (rowmin, rowmax, colmin, colmax) = get_rect_coordinates_from_roi(froi_signal)
    signal = img[rowmin:rowmax + 1, colmin:colmax + 1]
    if croi_background:
        (rowmin, rowmax, colmin, colmax) = croi_background
    elif froi_background:
        (rowmin, rowmax, colmin, colmax) = get_rect_coordinates_from_roi(froi_background)
    background = img[rowmin:rowmax + 1, colmin:colmax + 1]
    cnr_val = (signal.mean() - background.mean()) / background.std()
    return cnr_val

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def NRMSE(img, ref, mask='whole'):
    """Compute Normalized Root Mean Square Error."""
    from numpy.linalg import norm
    if img.shape != ref.shape:
        raise ValueError('Input arrays must have same shape.')
    if img.ndim != 2 or ref.ndim != 2:
        raise ValueError('Input arrays must be 2D.')
    if isinstance(mask, str):
        if mask == 'whole' or mask is None:
            vimg = img
            vref = ref
        elif mask == 'circ':
            (nrow, ncol) = img.shape
            mask_arr = get_circular_mask(nrow, ncol)
            vimg = img[mask_arr]
            vref = ref[mask_arr]
        else:
            raise ValueError('Invalid mask type.')
    elif isinstance(mask, np.ndarray):
        if mask.dtype is not np.dtype('bool'):
            raise ValueError('Mask must be boolean array.')
        vimg = img[mask]
        vref = ref[mask]
    else:
        raise ValueError('Invalid mask type.')
    vimg = vimg.astype(np.float64)
    vref = vref.astype(np.float64)
    val = norm(vimg - vref) / norm(vref)
    return val

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def SSIM(img1, img2, circ_crop=True, L=None, K1=0.01, K2=0.03, sigma=1.5, local_ssim=False):
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity as compare_ssim
    except ImportError:
        from skimage.measure import compare_ssim
    if img1.shape != img2.shape:
        raise ValueError('Input images must have same shape.')
    vimg1 = np.zeros(img1.shape)
    vimg2 = np.zeros(img2.shape)
    if circ_crop:
        (nrow, ncol) = img1.shape
        mask = get_circular_mask(nrow, ncol)
        vimg1[mask] = img1[mask]
        vimg2[mask] = img2[mask]
    else:
        vimg1 = img1
        vimg2 = img2
    val = compare_ssim(vimg1, vimg2, data_range=L, gaussian_weights=True, sigma=sigma, K1=K1, K2=K2, use_sample_covariance=False, full=local_ssim)
    return val

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_cpu_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'sinogram.tiff')
    output_dir = os.path.join(script_dir, 'output_fbp_cpu')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'--- FBP CPU Reconstruction (Headless) ---')
    if not os.path.exists(data_file):
        print(f'Error: Data file not found at {data_file}')
        return
    try:
        sino = read_tiff(data_file)
    except Exception as e:
        print(f'Error reading image: {e}')
        return
    print(f'Sinogram shape: {sino.shape}')
    pixel_size = 0.0029
    last_angle = 2 * np.pi
    n_angles = sino.shape[0]
    angles = np.linspace(0, last_angle, n_angles, endpoint=False)
    print('> Starting Reconstruction (CPU)...')
    try:
        rec = reconstruct(sino, angles, 'FBP', pixel_size=pixel_size)
    except Exception as e:
        print(f'Reconstruction failed: {e}')
        return
    output_filename = os.path.join(output_dir, 'recon_result.tiff')
    print(f'Saving result to: {output_filename}')
    if rec.ndim == 2:
        write_tiff(output_filename, rec)
    else:
        write_tiff_stack(os.path.join(output_dir, 'recon_result'), rec)
    print('Done.')
    print('\n--- Evaluation ---')
    roi_signal = os.path.join(script_dir, 'data', 'signal.roi')
    roi_background = os.path.join(script_dir, 'data', 'background.roi')
    if os.path.exists(roi_signal) and os.path.exists(roi_background):
        try:
            print('Generating Reference (Ground Truth) for evaluation...')
            sino_ref = read_tiff(data_file)
            try:
                ref = reconstruct(sino_ref, angles, 'SIRT_CUDA', parameters={'iterations': 200}, pixel_size=pixel_size)
            except:
                print('GPU Reference generation failed (or not available). Using CPU SIRT (50 iters) as reference...')
                ref = reconstruct(sino_ref, angles, 'SIRT', parameters={'iterations': 50}, pixel_size=pixel_size)
            ssim = SSIM(rec, ref)
            nrmse = NRMSE(rec, ref)
            cnr = CNR(rec, froi_signal=roi_signal, froi_background=roi_background)
            print(f'SSIM : {ssim:.4f}')
            print(f'NRMSE: {nrmse:.4f}')
            print(f'CNR  : {cnr:.4f}')
        except Exception as e:
            print(f'Evaluation failed: {e}')
    else:
        print('ROI files not found, skipping evaluation.')
if __name__ == '__main__':
    main()