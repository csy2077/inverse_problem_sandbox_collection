import os
import numpy as np
import tifffile
from read_roi import read_roi_file
from functools import reduce
import operator
import astra
from astra import data2d, projector, creators, algorithm, functions


# --- Extracted Dependencies ---

class OpTomo(scipy.sparse.linalg.LinearOperator):
    """Object that imitates a projection matrix with a given projector."""

    def __init__(self, proj_id):
        self.dtype = np.float32
        try:
            self.vg = projector.volume_geometry(proj_id)
            self.pg = projector.projection_geometry(proj_id)
            self.data_mod = data2d
            self.appendString = ""
            if projector.is_cuda(proj_id):
                self.appendString += "_CUDA"
        except Exception:
            from astra import data3d, projector3d
            self.vg = projector3d.volume_geometry(proj_id)
            self.pg = projector3d.projection_geometry(proj_id)
            self.data_mod = data3d
            self.appendString = "3D"
            if projector3d.is_cuda(proj_id):
                self.appendString += "_CUDA"

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

def read_tiff(fname):
    """Read a 2D TIFF image."""
    fname = os.path.abspath(fname)
    img = tifffile.imread(fname)
    return img

def write_tiff(fname, img, overwrite=True):
    """Write an array to a TIFF file."""
    if not (fname.endswith('.tif') or fname.endswith('.tiff')):
        fname = fname + '.tiff'
    tifffile.imsave(fname, img)
    print(f"File saved: {fname}")

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

    print(f'> Saving stack of images: {fname}_{"#" * digit}.tiff')
    for iz in range(nslice):
        outfile = fname + '_' + str(iz + start).zfill(digit) + '.tiff'
        tifffile.imsave(outfile, data[iz].astype(dtype))

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

    return rowmin, rowmax, colmin, colmax

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

def recon_stack(proj, method, pmat, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
    """Reconstruct a stack of sinograms or projections."""
    if parameters is None:
        parameters = {}

    if not sinogram_order:
        proj = np.swapaxes(proj, 0, 1)

    nslice, na, nd = proj.shape
    rec = np.zeros((nslice, nd, nd), dtype=np.float32)

    for s in range(nslice):
        print(f'\r  Reconstructing slice {s + 1}/{nslice}', end='')
        rec[s] = recon_slice(proj[s], method, pmat, parameters=parameters,
                             pixel_size=pixel_size, offset=offset)
    print()
    return rec

def reconstruct(tomo, angles, method, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
    """Reconstruct tomographic data using specified method."""
    if tomo.ndim != 2 and tomo.ndim != 3:
        raise ValueError('Invalid shape of array. Must have 2 or 3 dimensions.')

    nd = tomo.shape[-1]
    pmat = get_astra_proj_matrix(nd, angles, method)

    if tomo.ndim == 2:
        out = recon_slice(tomo, method, pmat, parameters=parameters,
                          pixel_size=pixel_size, offset=offset)
    elif tomo.ndim == 3:
        out = recon_stack(tomo, method, pmat, parameters=parameters,
                          pixel_size=pixel_size, offset=offset, sinogram_order=sinogram_order)
    else:
        raise ValueError('Invalid array dimensions. Must be 2D or 3D.')

    return out

def get_circular_mask(nrow, ncol, radius=None, center=None):
    """Get a boolean circular mask."""
    if radius is None:
        radius = min(ncol, nrow) / 2
    if center is None:
        yc = ncol / 2.0
        xc = nrow / 2.0
    else:
        yc, xc = center

    ny = np.arange(ncol)
    nx = np.arange(nrow)
    x, y = np.meshgrid(nx, ny)
    mask = ((y - yc + 0.5) ** 2 + (x - xc + 0.5) ** 2) < (radius) ** 2
    return mask

def CNR(img, croi_signal=[], croi_background=[], froi_signal=[], froi_background=[]):
    """Compute Contrast-to-Noise Ratio."""
    if img.ndim != 2:
        raise ValueError("Input array must have 2 dimensions.")

    if croi_signal:
        rowmin, rowmax, colmin, colmax = croi_signal
    if froi_signal:
        rowmin, rowmax, colmin, colmax = get_rect_coordinates_from_roi(froi_signal)

    signal = img[rowmin:(rowmax + 1), colmin:(colmax + 1)]

    if croi_background:
        rowmin, rowmax, colmin, colmax = croi_background
    elif froi_background:
        rowmin, rowmax, colmin, colmax = get_rect_coordinates_from_roi(froi_background)

    background = img[rowmin:(rowmax + 1), colmin:(colmax + 1)]
    cnr_val = (signal.mean() - background.mean()) / background.std()
    return cnr_val

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
            nrow, ncol = img.shape
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
        nrow, ncol = img1.shape
        mask = get_circular_mask(nrow, ncol)
        vimg1[mask] = img1[mask]
        vimg2[mask] = img2[mask]
    else:
        vimg1 = img1
        vimg2 = img2

    val = compare_ssim(vimg1, vimg2, data_range=L, gaussian_weights=True,
                       sigma=sigma, K1=K1, K2=K2, use_sample_covariance=False, full=local_ssim)
    return val

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'sinogram.tiff')
    output_dir = os.path.join(script_dir, 'output_fbp_cpu')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- FBP CPU Reconstruction (Headless) ---")

    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # Read sinogram (pre-processed data)
    try:
        sino = read_tiff(data_file)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    print(f"Sinogram shape: {sino.shape}")

    # Parameters
    pixel_size = 0.0029
    last_angle = 2 * np.pi
    n_angles = sino.shape[0]
    angles = np.linspace(0, last_angle, n_angles, endpoint=False)

    print('> Starting Reconstruction (CPU)...')
    # Use FBP (CPU)
    try:
        rec = reconstruct(sino, angles, 'FBP', pixel_size=pixel_size)
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        return

    output_filename = os.path.join(output_dir, 'recon_result.tiff')
    print(f"Saving result to: {output_filename}")

    if rec.ndim == 2:
        write_tiff(output_filename, rec)
    else:
        write_tiff_stack(os.path.join(output_dir, 'recon_result'), rec)

    print("Done.")

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    roi_signal = os.path.join(script_dir, 'data', 'signal.roi')
    roi_background = os.path.join(script_dir, 'data', 'background.roi')

    if os.path.exists(roi_signal) and os.path.exists(roi_background):
        try:
            print("Generating Reference (Ground Truth) for evaluation...")
            # Re-read sinogram for reference generation
            sino_ref = read_tiff(data_file)

            # Use SIRT_CUDA 200 iterations as reference if possible, else CPU SIRT
            try:
                ref = reconstruct(sino_ref, angles, 'SIRT_CUDA',
                                  parameters={'iterations': 200}, pixel_size=pixel_size)
            except:
                print("GPU Reference generation failed (or not available). Using CPU SIRT (50 iters) as reference...")
                ref = reconstruct(sino_ref, angles, 'SIRT',
                                  parameters={'iterations': 50}, pixel_size=pixel_size)

            ssim = SSIM(rec, ref)
            nrmse = NRMSE(rec, ref)
            cnr = CNR(rec, froi_signal=roi_signal, froi_background=roi_background)

            print(f"SSIM : {ssim:.4f}")
            print(f"NRMSE: {nrmse:.4f}")
            print(f"CNR  : {cnr:.4f}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    else:
        print("ROI files not found, skipping evaluation.")
