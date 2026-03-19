# ---------------------------------------------------------------------------------
# Standalone Tomographic NN-FBP (Neural Network Filtered Back Projection) Script
# ---------------------------------------------------------------------------------
# This script performs complete CT reconstruction using NN-FBP algorithm
# via ASTRA Toolbox. It's a self-contained version of the neutompy workflow.
#
# Problem: Tomographic Reconstruction from Limited/Sparse Sinogram Data
# Algorithm: Neural Network Filtered Back Projection (NN-FBP)
#
# Reference: Pelt, D. M., & Batenburg, K. J. (2013). Fast tomographic 
#            reconstruction from limited data using artificial neural networks.
#            IEEE Transactions on Image Processing, 22(12), 5238-5251.
#
# ---------------------------------------------------------------------------------
# FILE DEPENDENCIES:
# ---------------------------------------------------------------------------------
# Input Files:
#   - data/sinogram.tiff : Pre-processed sinogram data (2D TIFF image)
#                          Shape: (n_angles, n_detector_pixels)
#   - data/signal.roi    : ImageJ ROI file defining signal region (for CNR eval)
#   - data/background.roi: ImageJ ROI file defining background region (for CNR eval)
#
# Output Files:
#   - output_nnfbp/hqrecs/sample_XXXX.tiff    : High-quality FBP reconstructions 
#                                                (used as training targets)
#   - output_nnfbp/trainfiles/train_XXXXX.mat : Training data files (MAT format)
#   - output_nnfbp/filters.mat                : Trained NN-FBP filter weights
#   - output_nnfbp/recon-nnfbp/sample_XXXX.tiff: NN-FBP reconstructed slices
#
# Model Weights: 
#   - Filters are trained during execution and saved to output_nnfbp/filters.mat
#   - The training uses HQ reconstructions as ground truth
#
# Required Packages:
#   - numpy, tifffile, astra (ASTRA Toolbox), scipy, read_roi, numexpr
#   - skimage (for SSIM evaluation)
# ---------------------------------------------------------------------------------

import os
import sys
import errno
import glob
import random
import time
import numpy as np
import tifffile
from read_roi import read_roi_file
from functools import reduce
import operator
from scipy.signal import fftconvolve
import scipy.io as sio
import scipy.sparse as ss
import scipy.linalg as la
import numexpr

# Set CUDA device
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Import ASTRA after setting CUDA device
import astra
from astra import data2d, projector, creators, algorithm, functions
import scipy.sparse.linalg

# Try to import fblas for faster computation
try:
    import scipy.linalg.fblas as fblas
    hasfblas = True
except:
    hasfblas = False


# ===============================================================================
# Utility Functions
# ===============================================================================
def mkdir_p(path):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def sigmoid(x):
    """Sigmoid activation function."""
    return numexpr.evaluate("1./(1.+exp(-x))")


# ===============================================================================
# OpTomo class (from ASTRA Toolbox) - Projection matrix operator
# ===============================================================================
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
        if extraOptions and 'FilterType' in list(extraOptions.keys()):
            cfg['FilterType'] = extraOptions['FilterType']
            opts = {key: extraOptions[key] for key in extraOptions if key != 'FilterType'}
        else:
            opts = extraOptions if extraOptions else {}
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


# ===============================================================================
# Image I/O Functions
# ===============================================================================
def read_tiff(fname):
    """Read a 2D TIFF image."""
    fname = os.path.abspath(fname)
    img = tifffile.imread(fname)
    return img


def read_image(fname):
    """Read a 2D TIFF or FITS image."""
    if not os.path.isfile(fname):
        raise ValueError(f'No such file {fname}')
    if fname.endswith('.tif') or fname.endswith('.tiff') or fname.endswith('.TIF') or fname.endswith('.TIFF'):
        return read_tiff(fname)
    else:
        raise ValueError(f'File extension not valid: {fname}')


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


# ===============================================================================
# ROI Reading Functions
# ===============================================================================
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


# ===============================================================================
# Reconstruction Functions
# ===============================================================================
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


# ===============================================================================
# NN-FBP Custom Functions
# ===============================================================================
def customFBP(W, f, s):
    """
    Custom FBP reconstruction with specified filter.
    
    Parameters
    ----------
    W : OpTomo
        ASTRA projection matrix operator
    f : ndarray
        Filter to apply in frequency domain
    s : ndarray
        Sinogram data
        
    Returns
    -------
    rec : ndarray
        Reconstructed image
    """
    sf = np.zeros_like(s)
    padded = np.zeros(s.shape[1] * 2)
    l = int(s.shape[1] / 2.)
    r = l + s.shape[1]
    if len(f.shape) == 1:
        f = np.tile(f, (sf.shape[0], 1))
    for i in range(sf.shape[0]):
        padded[l:r] = s[i]
        padded[:l] = padded[l]
        padded[r:] = padded[r - 1]
        sf[i] = fftconvolve(padded, f[i], 'same')[l:r]
    return (W.T * sf).reshape(W.vshape)


# ===============================================================================
# NN-FBP Training Data Class
# ===============================================================================
class MATTrainingData:
    """Training data class that uses MAT files to store data."""

    def __init__(self, fls, dataname='mat'):
        self.fls = fls
        self.nBlocks = len(fls)
        self.dataname = dataname
        self.tileM = None
        self.nPar = (sio.loadmat(self.fls[0])[self.dataname]).shape[1] - 1

    def getDataBlock(self, i):
        if self.tileM is not None:
            data = sio.loadmat(self.fls[i])[self.dataname]
            data[:, 0:self.nPar] = 2 * (data[:, 0:self.nPar] - self.tileM) / self.maxmin - 1
            data[:, self.nPar] = 0.25 + (data[:, self.nPar] - self.minIn) / (2 * (self.maxIn - self.minIn))
            return data
        else:
            return sio.loadmat(self.fls[i])[self.dataname]

    def normalizeData(self, minL, maxL, minIn, maxIn):
        data = sio.loadmat(self.fls[0])[self.dataname]
        self.tileM = np.tile(minL, (data.shape[0], 1))
        self.maxmin = np.tile(maxL - minL, (data.shape[0], 1))
        self.minIn = minIn
        self.maxIn = maxIn

    def getMinMax(self):
        """Returns the minimum and maximum values of each column."""
        minL = np.empty(self.nPar)
        minL.fill(np.inf)
        maxL = np.empty(self.nPar)
        maxL.fill(-np.inf)
        maxIn = -np.inf
        minIn = np.inf
        for i in range(self.nBlocks):
            data = self.getDataBlock(i)
            if data is None:
                continue
            maxL = np.maximum(maxL, data[:, 0:self.nPar].max(0))
            minL = np.minimum(maxL, data[:, 0:self.nPar].min(0))
            maxIn = np.max([maxIn, data[:, self.nPar].max()])
            minIn = np.min([minIn, data[:, self.nPar].min()])
        return (minL, maxL, minIn, maxIn)


# ===============================================================================
# NN-FBP Network Class
# ===============================================================================
class Network:
    """
    Neural network for NN-FBP training using Levenberg-Marquardt method.
    """

    def __init__(self, nHiddenNodes, trainData, valData, setinit=None):
        self.tTD = trainData
        self.vTD = valData
        self.nHid = nHiddenNodes
        self.nIn = self.tTD.getDataBlock(0).shape[1] - 1
        self.jacDiff = np.zeros((self.nHid) * (self.nIn + 1) + self.nHid + 1)
        self.jac2 = np.zeros(((self.nHid) * (self.nIn + 1) + self.nHid + 1,
                              (self.nHid) * (self.nIn + 1) + self.nHid + 1))
        self.setinit = setinit

    def __inittrain(self):
        """Initialize training parameters."""
        self.l1 = 2 * np.random.rand(self.nIn + 1, self.nHid) - 1
        if self.setinit is not None:
            self.l1.fill(0)
            nd = self.nIn / self.setinit[0]
            for i, j in enumerate(self.setinit[1]):
                self.l1[j * nd:(j + 1) * nd, i] = 2 * np.random.rand(nd) - 1
                self.l1[-1, i] = 2 * np.random.rand(1) - 1
        beta = 0.7 * self.nHid ** (1. / (self.nIn))
        l1norm = np.linalg.norm(self.l1)
        self.l1 *= beta / l1norm
        self.l2 = 2 * np.random.rand(self.nHid + 1) - 1
        self.l2 /= np.linalg.norm(self.l2)
        self.minl1 = self.l1.copy()
        self.minl2 = self.l2.copy()
        self.minmax = self.tTD.getMinMax()
        self.tTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.vTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.ident = np.eye((self.nHid) * (self.nIn + 1) + self.nHid + 1)

    def __processDataBlock(self, data):
        """Process a block of data through the network."""
        valOut = data[:, -1].copy()
        data[:, -1] = -np.ones(data.shape[0])
        hiddenOut = np.empty((data.shape[0], self.l1.shape[1] + 1))
        hiddenOut[:, 0:self.l1.shape[1]] = sigmoid(np.dot(data, self.l1))
        hiddenOut[:, -1] = -1
        rawVals = np.dot(hiddenOut, self.l2)
        vals = sigmoid(rawVals)
        return vals, valOut, hiddenOut

    def __getTSE(self, dat):
        """Returns the total squared error of a data block."""
        tse = 0.
        for i in range(dat.nBlocks):
            data = dat.getDataBlock(i)
            vals, valOut, hiddenOut = self.__processDataBlock(data)
            tse += numexpr.evaluate('sum((vals - valOut)**2)')
        return tse

    def __setJac2(self):
        """Calculates J^T J and J^T e for the training data."""
        self.jac2.fill(0)
        self.jacDiff.fill(0)
        for i in range(self.tTD.nBlocks):
            data = self.tTD.getDataBlock(i)
            vals, valOut, hiddenOut = self.__processDataBlock(data)
            diffs = numexpr.evaluate('valOut - vals')
            jac = np.empty((data.shape[0], (self.nHid) * (self.nIn + 1) + self.nHid + 1))
            d0 = numexpr.evaluate('-vals * (1 - vals)')
            ot = (np.outer(d0, self.l2))
            dj = numexpr.evaluate('hiddenOut * (1 - hiddenOut) * ot')
            I = np.tile(np.arange(data.shape[0]), (self.nHid + 1, 1)).flatten('F')
            J = np.arange(data.shape[0] * (self.nHid + 1))
            Q = ss.csc_matrix((dj.flatten(), np.vstack((J, I))),
                              (data.shape[0] * (self.nHid + 1), data.shape[0]))
            jac[:, 0:self.nHid + 1] = ss.spdiags(d0, 0, data.shape[0], data.shape[0]).dot(hiddenOut)
            Q2 = np.reshape(Q.dot(data), (data.shape[0], (self.nIn + 1) * (self.nHid + 1)))
            jac[:, self.nHid + 1:jac.shape[1]] = Q2[:, 0:Q2.shape[1] - (self.nIn + 1)]
            if hasfblas:
                self.jac2 += fblas.dgemm(1.0, a=jac.T, b=jac.T, trans_b=True)
                self.jacDiff += fblas.dgemv(1.0, a=jac.T, x=diffs)
            else:
                self.jac2 += np.dot(jac.T, jac)
                self.jacDiff += np.dot(jac.T, diffs)

    def train(self):
        """Train the network using Levenberg-Marquardt method."""
        self.__inittrain()
        mu = 100000.
        muUpdate = 10
        prevValError = np.Inf
        bestCounter = 0
        tse = self.__getTSE(self.tTD)
        self.allls = []
        
        for i in range(1000000):
            self.__setJac2()
            try:
                dw = -la.cho_solve(la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
            except la.LinAlgError:
                break
            done = -1
            while done <= 0:
                self.l2 += dw[0:self.nHid + 1]
                for k in range(self.nHid):
                    start = self.nHid + 1 + k * (self.nIn + 1)
                    if self.setinit is not None:
                        nd = self.nIn / self.setinit[0]
                        j = self.setinit[1][k]
                        self.l1[j * nd:(j + 1) * nd, k] += dw[start + j * nd:start + (j + 1) * nd]
                        self.l1[-1, k] += dw[start + self.nIn]
                    else:
                        self.l1[:, k] += dw[start:start + self.nIn + 1]
                newtse = self.__getTSE(self.tTD)
                if newtse < tse:
                    if done == -1:
                        mu /= muUpdate
                    if mu <= 1e-100:
                        mu = 1e-99
                    done = 1
                else:
                    done = 0
                    mu *= muUpdate
                    if mu >= 1e20:
                        done = 2
                        break
                    self.l2 -= dw[0:self.nHid + 1]
                    for k in range(self.nHid):
                        start = self.nHid + 1 + k * (self.nIn + 1)
                        if self.setinit is not None:
                            nd = self.nIn / self.setinit[0]
                            j = self.setinit[1][k]
                            self.l1[j * nd:(j + 1) * nd, k] -= dw[start + j * nd:start + (j + 1) * nd]
                            self.l1[-1, k] -= dw[start + self.nIn]
                        else:
                            self.l1[:, k] -= dw[start:start + self.nIn + 1]
                    try:
                        dw = -la.cho_solve(la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
                    except la.LinAlgError:
                        done = 2
            gradSize = np.linalg.norm(self.jacDiff)
            if done == 2:
                break
            curValErr = self.__getTSE(self.vTD)
            if curValErr > prevValError:
                bestCounter += 1
            else:
                prevValError = curValErr
                self.minl1 = self.l1.copy()
                self.minl2 = self.l2.copy()
                if (newtse / tse < 0.999):
                    bestCounter = 0
                else:
                    bestCounter += 1
            if bestCounter == 50:
                break
            if gradSize < 1e-8:
                break
            tse = newtse
            print(f'  Training iteration {i}, validation error: {prevValError:.6f}')
            self.allls.append([self.minl1, self.minl2])
        self.l1 = self.minl1
        self.l2 = self.minl2
        self.valErr = prevValError

    def saveToDisk(self, fn):
        """Save trained network to disk."""
        sio.savemat(fn, {'l1': self.l1, 'l2': self.l2, 'minmax': self.minmax}, do_compression=True)


# ===============================================================================
# NN-FBP Plugin Functions
# ===============================================================================
def nnfbp_prepare_training_data(sinogram, hq_rec, z_id, traindir, pmat, npick=100, nlinear=2):
    """
    Prepare training data for NN-FBP.
    
    Parameters
    ----------
    sinogram : ndarray
        Input sinogram (2D array)
    hq_rec : ndarray
        High-quality reconstruction used as target
    z_id : int
        Slice index
    traindir : str
        Directory to save training files
    pmat : OpTomo
        ASTRA projection matrix operator
    npick : int
        Number of random pixels to pick
    nlinear : int
        Number of linear steps in exponential binning
    """
    mkdir_p(traindir)
    
    W = pmat
    s = sinogram.astype(np.float32)
    v_shape = W.vshape
    
    # Create basis filters
    fs = s.shape[1]
    if fs % 2 == 0:
        fs += 1
    mf = int(fs / 2)
    w = 1
    c = mf
    bas = np.zeros(fs, dtype=np.float32)
    basis = []
    count = 0
    while c < fs:
        bas[:] = 0
        l = c
        r = c + w
        if r > fs:
            r = fs
        bas[l:r] = 1
        if l != 0:
            l = fs - c - w
            r = l + w
            if l < 0:
                l = 0
            bas[l:r] = 1
        basis.append(bas.copy())
        c += w
        count += 1
        if count > nlinear:
            w = 2 * w
    
    nf = len(basis)
    
    # Create background mask
    bck = (W.T * np.ones_like(s) < s.shape[0] - 0.5).reshape(v_shape)
    
    # Pick random pixels
    out = np.zeros((npick, nf + 1))
    pl = np.random.random((npick, 2))
    pl[:, 0] *= v_shape[0]
    pl[:, 1] *= v_shape[1]
    pl = pl.astype(np.int32)
    
    for i in range(npick):
        while bck[pl[i, 0], pl[i, 1]]:
            pl[i, 0] = int(np.random.random(1) * v_shape[0])
            pl[i, 1] = int(np.random.random(1) * v_shape[1])
    
    out[:, -1] = hq_rec[pl[:, 0], pl[:, 1]]
    
    for i, bas in enumerate(basis):
        img = customFBP(W, bas, s)
        out[:, i] = img[pl[:, 0], pl[:, 1]]
    
    outfn = os.path.join(traindir, f"train_{z_id:05d}.mat")
    sio.savemat(outfn, {'mat': out}, do_compression=True)


def nnfbp_train(traindir, nhid, filter_file, val_rat=0.5):
    """
    Train NN-FBP filters.
    
    Parameters
    ----------
    traindir : str
        Directory containing training files
    nhid : int
        Number of hidden nodes
    filter_file : str
        File to save trained filters
    val_rat : float
        Fraction of training examples to use as validation
    """
    fls = glob.glob(os.path.join(traindir, '*.mat'))
    random.shuffle(fls)
    nval = int(val_rat * len(fls))
    val = MATTrainingData(fls[:nval])
    trn = MATTrainingData(fls[nval:])
    n = Network(nhid, trn, val)
    n.train()
    n.saveToDisk(filter_file)


def nnfbp_reconstruct(sinogram, filter_file, pmat, nlinear=2):
    """
    Reconstruct using NN-FBP.
    
    Parameters
    ----------
    sinogram : ndarray
        Input sinogram (2D array)
    filter_file : str
        File with trained filters
    pmat : OpTomo
        ASTRA projection matrix operator
    nlinear : int
        Number of linear steps in exponential binning
        
    Returns
    -------
    rec : ndarray
        Reconstructed image
    """
    W = pmat
    s = sinogram.astype(np.float32)
    v = np.zeros(W.vshape, dtype=np.float32)
    
    # Create basis filters
    fs = s.shape[1]
    if fs % 2 == 0:
        fs += 1
    mf = int(fs / 2)
    w = 1
    c = mf
    bas = np.zeros(fs, dtype=np.float32)
    basis = []
    count = 0
    while c < fs:
        bas[:] = 0
        l = c
        r = c + w
        if r > fs:
            r = fs
        bas[l:r] = 1
        if l != 0:
            l = fs - c - w
            r = l + w
            if l < 0:
                l = 0
            bas[l:r] = 1
        basis.append(bas.copy())
        c += w
        count += 1
        if count > nlinear:
            w = 2 * w
    
    nf = len(basis)
    
    # Load trained filters
    fl = sio.loadmat(filter_file)
    l1 = fl['l1']
    l2 = fl['l2'].transpose()
    minmax = fl['minmax'][0]
    minL = minmax[0]
    maxL = minmax[1]
    minIn = minmax[2]
    maxIn = minmax[3]
    
    mindivmax = minL / (maxL - minL)
    mindivmax[np.isnan(mindivmax)] = 0
    mindivmax[np.isinf(mindivmax)] = 0
    divmaxmin = 1. / (maxL - minL)
    divmaxmin[np.isnan(divmaxmin)] = 0
    divmaxmin[np.isinf(divmaxmin)] = 0
    
    nHid = l1.shape[1]
    nsl = 1
    dims = [nHid, nsl]
    dims.extend(basis[0].shape)
    filters = np.empty(dims)
    offsets = np.empty(nHid)
    
    for i in range(nHid):
        wv = (2 * l1[0:l1.shape[0] - 1, i] * divmaxmin).transpose()
        filters[i] = np.zeros(dims[1:])
        for t, bas in enumerate(basis):
            for l in range(nsl):
                filters[i, l] += wv[t + l * len(basis)] * bas
        offsets[i] = 2 * np.dot(l1[0:l1.shape[0] - 1, i], mindivmax.transpose()) + np.sum(l1[:, i])
    
    # Reconstruct
    v[:] = 0
    for i in range(l2.shape[0] - 1):
        mult = float(l2[i])
        offs = float(offsets[i])
        back = customFBP(W, filters[i, 0], s)
        v[:] += numexpr.evaluate('mult/(1.+exp(-(back-offs)))')
    
    v[:] = sigmoid(v - l2[-1])
    v[:] = (v - 0.25) * 2 * (maxIn - minIn) + minIn
    
    return v


# ===============================================================================
# Metric Functions
# ===============================================================================
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


# ===============================================================================
# Main Function
# ===============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'data', 'sinogram.tiff')
    
    # Setup output folders
    base_output_dir = os.path.join(script_dir, 'output_nnfbp')
    hqrec_folder = os.path.join(base_output_dir, 'hqrecs/')
    nnfbp_rec_folder = os.path.join(base_output_dir, 'recon-nnfbp/')
    train_dir = os.path.join(base_output_dir, 'trainfiles/')
    
    for d in [hqrec_folder, nnfbp_rec_folder, train_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Configuration
    hidden_nodes = 3
    npick = 5000  # Reduced number of pixels for faster execution
    filter_file = os.path.join(base_output_dir, 'filters.mat')

    print(f"--- NN-FBP Reconstruction (Headless) ---")
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # Read sinogram
    try:
        sino_2d = read_tiff(data_file)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    print(f"Sinogram 2D shape: {sino_2d.shape}")
    
    # Create a fake 3D stack (Angles, Slices, Pixels) for NN-FBP training/testing
    # 10 slices total: 0-4 for training, 5-9 for testing
    n_slices = 10
    n_angles, n_pixels = sino_2d.shape
    
    print(f"Creating fake 3D stack with {n_slices} slices...")
    norm = np.zeros((n_angles, n_slices, n_pixels), dtype=sino_2d.dtype)
    for i in range(n_slices):
        norm[:, i, :] = sino_2d
        
    print(f"Stack shape: {norm.shape}")

    pixel_size = 0.0029
    last_angle = 2 * np.pi
    angles = np.linspace(0, last_angle, n_angles, endpoint=False)

    # --- 1. High-quality reconstruction (Training Data Generation) ---
    train_slice_start = 0
    train_slice_end = 4
    
    print("Generating HQ training data (FBP)...")
    # Use FBP_CUDA for ground truth/HQ
    try:
        rec = reconstruct(norm[:, train_slice_start:train_slice_end + 1, :], angles, 'FBP_CUDA',
                         parameters={"FilterType": "hamming"}, pixel_size=pixel_size)
    except Exception as e:
        print(f"GPU FBP failed: {e}. Falling back to CPU.")
        rec = reconstruct(norm[:, train_slice_start:train_slice_end + 1, :], angles, 'FBP',
                         parameters={"FilterType": "hamming"}, pixel_size=pixel_size)
    
    # Save HQ recs
    print(f"Saving HQ reconstructions to {hqrec_folder}...")
    write_tiff_stack(os.path.join(hqrec_folder, 'sample'), rec)

    # --- 2. NN-FBP training ---
    print("Training NN-FBP...")
    skip = 3  # reduction factor
    norm_train = norm[::skip, train_slice_start:train_slice_end + 1, :]
    angles_sparse = angles[::skip]
    
    # Get projection matrix for training
    nd = norm_train.shape[-1]
    pmat_train = get_astra_proj_matrix(nd, angles_sparse, 'NN-FBP')
    
    # Prepare training data for each slice
    print("> Preparing training data...")
    for z_id in range(train_slice_end - train_slice_start + 1):
        print(f"  Processing slice {z_id + 1}/{train_slice_end - train_slice_start + 1}")
        sino_slice = norm_train[:, z_id, :].astype(np.float32) / pixel_size
        hq_rec = rec[z_id]
        nnfbp_prepare_training_data(sino_slice, hq_rec, z_id, train_dir, pmat_train, npick=npick)
    
    # Train network
    print("> Training neural network...")
    nnfbp_train(train_dir, hidden_nodes, filter_file)
    print(f"Filters saved to: {filter_file}")

    # --- 3. NN-FBP reconstruction (Testing) ---
    test_slice_start = 5
    test_slice_end = 9
    print("Reconstructing with NN-FBP...")
    norm_test = norm[::skip, test_slice_start:test_slice_end + 1, :]
    
    # Get projection matrix for reconstruction
    pmat_rec = get_astra_proj_matrix(nd, angles_sparse, 'NN-FBP')
    
    # Reconstruct each test slice
    n_test_slices = test_slice_end - test_slice_start + 1
    rec_nnfbp = np.zeros((n_test_slices, nd, nd), dtype=np.float32)
    
    for z_id in range(n_test_slices):
        print(f"  Reconstructing slice {z_id + 1}/{n_test_slices}")
        sino_slice = norm_test[:, z_id, :].astype(np.float32) / pixel_size
        rec_nnfbp[z_id] = nnfbp_reconstruct(sino_slice, filter_file, pmat_rec)

    # Write NN-FBP reconstructed images
    print(f"Saving NN-FBP results to {nnfbp_rec_folder}...")
    write_tiff_stack(os.path.join(nnfbp_rec_folder, 'sample'), rec_nnfbp)
    print("Done.")

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    roi_signal = os.path.join(script_dir, 'data', 'signal.roi')
    roi_background = os.path.join(script_dir, 'data', 'background.roi')
    
    if os.path.exists(roi_signal) and os.path.exists(roi_background):
        try:
            print("Generating Reference (Ground Truth) for evaluation...")
            # Use SIRT_CUDA 200 iterations as reference
            try:
                ref = reconstruct(sino_2d, angles, 'SIRT_CUDA',
                                 parameters={'iterations': 200}, pixel_size=pixel_size)
            except:
                print("GPU Reference generation failed. Using CPU SIRT (50 iters) as reference...")
                ref = reconstruct(sino_2d, angles, 'SIRT',
                                 parameters={'iterations': 50}, pixel_size=pixel_size)

            # Take the first slice of NN-FBP result for comparison
            rec_to_eval = rec_nnfbp[0]

            ssim = SSIM(rec_to_eval, ref)
            nrmse = NRMSE(rec_to_eval, ref)
            cnr = CNR(rec_to_eval, froi_signal=roi_signal, froi_background=roi_background)
            
            print(f"SSIM : {ssim:.4f}")
            print(f"NRMSE: {nrmse:.4f}")
            print(f"CNR  : {cnr:.4f}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    else:
        print("ROI files not found, skipping evaluation.")


if __name__ == "__main__":
    main()

