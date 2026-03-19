# Standalone Self-Contained Multi-Reflection BCDI Solver for SiC Crystal Reconstruction

# This script is a self-contained version of the Multi-Reflection Bragg Coherent Diffraction
# Imaging (BCDI) optimization solver. It combines all necessary modules from the mrbcdi package
# into a single file.

# Problem: SiC (Silicon Carbide) crystal reconstruction from 6 Bragg peaks
# Algorithm: Multi-Reflection BCDI with TV regularization and Adam optimizer

# Original Authors:
#     Siddharth Maddali
#     Argonne National Laboratory
#     smaddali@anl.gov

# Assembled into standalone form for portability.

# ================================================================================
#                               FILE DEPENDENCIES
# ================================================================================

# INPUT FILES:
#     - Data file: ./data/SiC_400nm.h5
#       (HDF5 file containing Bragg diffraction patterns and experimental metadata)
#       Contains:
#         - global/: Global experimental parameters (lattice parameters, crystal rotation)
#         - scans/: Individual Bragg peak measurements with:
#             - signal: Diffraction intensity data
#             - peak: Reciprocal space peak position (Angstrom^-1)
#             - miller_idx: Miller indices of the reflection
#             - RtoBragg: Rotation matrix to Bragg frame
#             - Bdet: Detector basis
#             - Breal: Real-space basis (nm)

# OUTPUT FILES:
#     - Reconstruction results: ./reconstructions/Results_SiC.h5
#       Contains:
#         - real_space/electron_density: Reconstructed 3D electron density
#         - real_space/lattice_deformation: 3D lattice displacement field (ux, uy, uz)
#         - reconstruction_error: Loss function history
#         - reciprocal_space/<scan>/wave: Reconstructed complex wave in reciprocal space
#         - reciprocal_space/<scan>/data: Measured diffraction data

#     - Plot files (PNG):
#         - ./reconstructions/SiC_error_phase1.png: Loss function after phase 1 optimization
#         - ./reconstructions/SiC_error_phase2.png: Loss function after phase 2 optimization  
#         - ./reconstructions/SiC_amplitude.png: Reconstructed electron density slices (XY, YZ, ZX)
#         - ./reconstructions/SiC_displacement.png: Lattice displacement field components
#         - ./reconstructions/SiC_diffraction_<n>.png: Measured vs. reconstructed diffraction patterns

# MODEL WEIGHTS:
#     - None (optimization-based reconstruction, no pre-trained models)

# DEPENDENCIES (Python packages):
#     - numpy
#     - h5py
#     - matplotlib
#     - torch (with CUDA support)
#     - scipy
#     - tqdm
#     - logzero

# ================================================================================
#                          HYPERPARAMETER MODIFICATIONS
# ================================================================================
#
# The optimization hyperparameters have been REDUCED for ultra-fast sanity check (~1000x speedup).
# Original values are commented out in the code; ultra-fast values are active.
#
# PHASE 1 (optim_plan):
#   Original:
#     minibatches: [800, 640, 480, 320, 160, 80, 40, 1]
#     minibatch_size: [4, 4, 4, 5, 5, 5, 5, 6]
#     iterations_per_minibatch: [6, 12, 25, 50, 100, 200, 400, 2000]
#     Total iterations: ~90,480
#
#   Ultra-fast (ACTIVE):
#     minibatches: [1, 1, 1, 1, 1, 1, 1, 1]       (minimal)
#     minibatch_size: [4, 4, 4, 5, 5, 5, 5, 6]    (unchanged)
#     iterations_per_minibatch: [1, 1, 1, 1, 1, 1, 1, 20]
#     Total iterations: ~27
#
# PHASE 2 (new_plan):
#   Original:
#     iterations_per_minibatch: [1000]
#
#   Ultra-fast (ACTIVE):
#     iterations_per_minibatch: [20]  (~50x fewer)
#
# Total: ~47 iterations (~1000x faster than original ~91,480)
#
# To restore original hyperparameters, swap the commented blocks in main().
#
# ================================================================================


import os
import sys
import time
import random
import itertools
import functools as ftools

import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t

from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation
from numpy.fft import fftshift as fftshift_o, fftn as fftn_o

from tqdm.auto import tqdm
from logzero import logger

# Set CUDA device to 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


#######################################################################
#                         TilePlot Module
#######################################################################

def TilePlot(images, layout, figsize=(10, 10), **kwargs):
    """
    Creates a tiled plot of multiple images.
    
    Args:
        images: tuple of 2D arrays to plot
        layout: (rows, cols) tuple for grid layout
        figsize: figure size tuple
        **kwargs: additional arguments (log_norm, color_scales, etc.)
    
    Returns:
        fig, im, ax: figure, image objects, and axes
    """
    rows, cols = layout
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single axis case
    if rows * cols == 1:
        ax = np.array([ax])
    else:
        ax = ax.flatten()
    
    im = []
    log_norm = kwargs.get('log_norm', False)
    
    for i, img in enumerate(images):
        if i < len(ax):
            # Check if image is complex
            if np.iscomplexobj(img):
                img = np.abs(img)
            
            if log_norm:
                from matplotlib.colors import LogNorm
                # Avoid log(0)
                img_plot = np.where(img > 0, img, img[img > 0].min() if np.any(img > 0) else 1e-10)
                im.append(ax[i].imshow(img_plot, norm=LogNorm()))
            else:
                im.append(ax[i].imshow(img))
    
    return fig, im, ax


#######################################################################
#                         Resample Module
#######################################################################

def getBufferSize(coef_of_exp, shp):
    """
    Calculate buffer size for resampling operations.
    """
    bufsize = np.array(
        [
            np.round(0.5 * n * (1./c - 1.)).astype(int)
            for n, c in zip(shp, coef_of_exp)
        ]
    )
    return bufsize


#######################################################################
#                         FFT Module
#######################################################################

def fft1d(arr, n):
    """1D FFT along axis n with shifts."""
    return fftshift_t(
        fftn_t(
            fftshift_t(arr, dim=[n]),
            dim=[n],
            norm='ortho'
        ),
        dim=[n]
    )


def ifft1d(arr, n):
    """1D inverse FFT along axis n with shifts."""
    return fftshift_t(
        ifftn_t(
            fftshift_t(arr, dim=[n]),
            dim=[n],
            norm='ortho'
        ),
        dim=[n]
    )


#######################################################################
#                         Grid Plugin
#######################################################################

class GridPlugin:
    """Plugin for setting up computational grids."""
    
    def setUpGrid(self):
        grid_np = np.mgrid[
            (-self._domainSize[0]//2.):(self._domainSize[0]//2.),
            (-self._domainSize[1]//2.):(self._domainSize[1]//2.),
            (-self._domainSize[2]//2.):(self._domainSize[2]//2.)
        ]
        self._gridprod = [
            torch.from_numpy(
                fftshift_o(
                    2. * np.pi * grid_np[(n+2)%3] * grid_np[(n+1)%3] / self._domainSize[n]
                )
            ).cuda()
            for n in range(3)
        ]
        return


#######################################################################
#                         Mount Plugin
#######################################################################

class MountPlugin(GridPlugin):
    """Plugin for mounting objects in diffractometer coordinates."""
    
    def __init__(self, size, R, Br, s0):
        self._domainSize = tuple(size for n in range(3))
        self.initializeDiffractometer(R, Br, s0)
        return

    def initializeDiffractometer(self, R, Br, s0):
        self._delTheta = np.pi/4.
        self._axis_dict = {'X': 0, 'Y': 1, 'Z': 2}
        self.createBuffer(Br, s0)
        self.setUpGrid()  # inherited from GridPlugin
        self.createMask()
        self.getRotationParameters(R, 'XYZ')
        self.prepareToShear(Br)
        return

    def createBuffer(self, Br, s0):
        self._diffbuff = getBufferSize(np.diag(Br)/s0, self._domainSize)
        self._padder = torch.nn.ConstantPad3d(
            list(itertools.chain(
                *[[n, n] if n > 0 else [0, 0] for n in self._diffbuff[::-1]]
            )), 0
        )
        self._skin = tuple(
            (0, N) if db <= 0 else (db, N+db)
            for db, N in zip(self._diffbuff, self._domainSize)
        )
        return

    def createMask(self):
        mask = np.ones(self._padder(torch.zeros(self._domainSize)).shape)
        for num in self._diffbuff:
            if num < 0:
                mask[-num:num, :, :] = 0.
            mask = np.transpose(mask, np.roll([0, 1, 2], -1))
        self._mask = torch.from_numpy(1. - mask).cuda()
        return

    def shearPrincipal(self, ax, shift):
        self._rho = torch.fft.ifftn(
            shift * torch.fft.fftn(self._rho, dim=[ax], norm='ortho'),
            dim=[ax],
            norm='ortho'
        )
        return

    def rotatePrincipal(self, angle, ax=2):
        ax1 = (ax+1) % 3
        ax2 = (ax+2) % 3
        shift1, shift2 = tuple(
            torch.exp(1.j * ang * self._gridprod[ax])
            for ang in [np.tan(angle/2.), -np.sin(angle)]
        )
        for x, sh in zip([ax1, ax2, ax1], [shift1, shift2, shift1]):
            self.shearPrincipal(x, sh)
        return

    def prepareToShear(self, Br):
        self._Bshear = Br.T / np.diag(Br).reshape(1, -1).repeat(3, axis=0)
        self._shearShift = []
        for n in range(3):
            np1, np2 = (n+1) % 3, (n+2) % 3
            alpha, beta = self._Bshear[np1, n], self._Bshear[np2, n]
            ax1, ax2 = (n-1) % 3, (n+1) % 3
            shift_arg = alpha*self._gridprod[ax1] + beta*self._gridprod[ax2]
            shift = torch.exp(1.j * shift_arg)
            self._shearShift.append(shift)
        return

    def shearResampleObject(self):
        for ax in range(3):
            self.shearPrincipal(ax, self._shearShift[ax])
        return

    def splitAngle(self, angle):
        anglist = [np.sign(angle)*self._delTheta] * int(np.absolute(angle//self._delTheta))
        if angle < 0.:
            if len(anglist) > 0:
                anglist[-1] += (angle % self._delTheta)
            else:
                anglist.append(angle % self._delTheta)
        else:
            anglist.append(angle % self._delTheta)
        return anglist

    def getRotationParameters(self, R, euler_convention='XYZ'):
        convention = [self._axis_dict[k] for k in list(euler_convention)]
        eulers = Rotation.from_matrix(R).as_euler(euler_convention)
        self._axes = []
        self._eulers = []
        for ang, ax in zip(eulers[::-1], convention[::-1]):
            anglist = self.splitAngle(ang)
            self._axes.extend([ax]*len(anglist))
            self._eulers.extend(anglist)
        return

    def rotateObject(self):
        for ang, ax in zip(self._eulers, self._axes):
            self.rotatePrincipal(ang, ax)
        return

    def bulkResampleObject(self):
        self._rhopad = self._padder(fftshift_t(self._rho))  # padded object array

        n = self._diffbuff[0]
        if n < 0:
            self._rhopad[-n:n, :, :] = ifft1d(
                (self._mask * fft1d(self._rhopad, 0))[-n:n, :, :],
                0
            )
        elif n > 0:
            self._rhopad[n:-n, :, :] = fft1d(self._rhopad[n:-n, :, :], 0)
            self._rhopad = ifft1d(self._rhopad, 0)

        n = self._diffbuff[1]
        if n < 0:
            self._rhopad[:, -n:n, :] = ifft1d(
                (self._mask * fft1d(self._rhopad, 1))[:, -n:n, :],
                1
            )
        elif n > 0:
            self._rhopad[:, n:-n, :] = fft1d(self._rhopad[:, n:-n, :], 1)
            self._rhopad = ifft1d(self._rhopad, 1)

        n = self._diffbuff[2]
        if n < 0:
            self._rhopad[:, :, -n:n] = ifft1d(
                (self._mask * fft1d(self._rhopad, 2))[:, :, -n:n],
                2
            )
        elif n > 0:
            self._rhopad[:, :, n:-n] = fft1d(self._rhopad[:, :, n:-n], 2)
            self._rhopad = ifft1d(self._rhopad, 2)

        self.removeBuffer()
        return

    def removeBuffer(self):
        self._rho = fftshift_t(
            self._rhopad[
                self._skin[0][0]:self._skin[0][1],
                self._skin[1][0]:self._skin[1][1],
                self._skin[2][0]:self._skin[2][1]
            ]
        )
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


#######################################################################
#                         Lattice Plugin
#######################################################################

class LatticePlugin:
    """Plugin for calculations involving 3D Bravais lattices."""

    def getLatticeBases(self, a, b, c, al, bt, gm):
        a, b, c = tuple(lat/10. for lat in [a, b, c])  # converting from Angstrom to nanometers
        al, bt, gm = tuple(np.pi/180. * ang for ang in [al, bt, gm])  # converting from degrees to radians
        p = (np.cos(al) - np.cos(bt)*np.cos(gm)) / (np.sin(bt)*np.sin(gm))
        q = np.sqrt(1. - p**2)
        self.basis_real = np.array(
            [
                [a, b*np.cos(gm), c*np.cos(bt)],
                [0., b*np.sin(gm), c*p*np.sin(bt)],
                [0., 0., c*q*np.sin(bt)]
            ]
        )
        if np.linalg.det(self.basis_real) < 0.:  # convert to right-handed
            self.basis_real[-1, -1] *= -1.

        self.basis_real = torch.from_numpy(self.basis_real).cuda()  # used for unit cell modulo
        self.basis_reciprocal = torch.linalg.inv(self.basis_real.T)
        self.metricTensor = self.basis_real.T @ self.basis_real
        self.planeStepTensor = torch.linalg.inv(self.metricTensor)
        self.d_jumps_unitcell = 1. / torch.sqrt(self.planeStepTensor.sum(axis=0))
        return

    def getPlaneSeparations(self):
        idx_arr = torch.from_numpy(np.array([list(idx) for idx in self.miller_idx]).T).cuda()
        d = 1. / torch.sqrt((idx_arr * (self.planeStepTensor @ idx_arr)).sum(axis=0))
        return d

    def getRotatedCrystalBases(self):
        R = torch.from_numpy(self.database['global'].attrs['Rcryst']).cuda()
        self.basis_real = R @ self.basis_real
        self.basis_reciprocal = R @ self.basis_reciprocal
        return


#######################################################################
#                         Object Plugin
#######################################################################

class ObjectPlugin:
    """Routines to create tight-bound object in array."""

    def createObjectBB_full(self, n_scan):
        """
        Creates phase object within bounding box from FULL set of optimizable variables.
        """
        self.scl = self.x[-len(self.bragg_condition):][n_scan]
        self.__amp__ = 0.5*(1. + torch.tanh(self.x[:self.N] / self.activ_a))
        self.__u__ = self.x[self.N:(4*self.N)].reshape(3, -1)
        phs = self.peaks[n_scan] @ self.__u__
        objBB = self.scl * self.__amp__ * torch.exp(2.j * np.pi * phs)
        return objBB.reshape(self.cubeSize, self.cubeSize, self.cubeSize)

    def buildFullBB(self):
        arr1 = [val*np.ones(self.N) for val in [-10.*self.activ_a, 0., 0., 0.]]
        arr1.append(np.zeros(len(self.bragg_condition)))
        self.x = torch.tensor(
            np.concatenate(tuple(arr1))
        ).cuda()  # this is the new array used to build the bound-box object
        self.x[self.these_only] = self.x_new  # slice optimized variables into original-sized array
        return

    def createObjectBB_part(self, n_scan):
        """
        Creates phase object within bounding box from RESTRICTED set of optimizable variables.
        """
        self.buildFullBB()
        objBB = self.createObjectBB_full(n_scan)
        return objBB


#######################################################################
#                         Optimizer Plugin
#######################################################################

def TV3D(arr):
    """
    Calculates total variation regularization of input 3D array,
    as the sum of total variation along each axis.
    """
    return torch.mean(
        torch.ravel(torch.abs(arr[1:, :, :] - arr[:-1, :, :])) +
        torch.ravel(torch.abs(arr[:, 1:, :] - arr[:, :-1, :])) +
        torch.ravel(torch.abs(arr[:, :, 1:] - arr[:, :, :-1]))
    )


class OptimizerPlugin:
    """Plugin for optimizer methods for the multi-reflection BCDI problem."""

    def initializeOptimizer(self,
                            optim_var,  # optimize over these variables
                            learning_rate, lambda_tv,
                            minibatch_plan, default_iterations=3000):
        """
        minibatch_plan format:

        {
            'minibatches':[ list of <N> ]
            'minibatch_size':[ list of <N> ]
            'iterations_per_minibatch':[ list of <N> ]
        }

        The lists range over the number of epochs. A singleton list means only one epoch.
        Minibatches are automatically and randomly generated.
        """
        self.lr = learning_rate
        try:
            dummy = len(self.error)  # do nothing new if this array already exists
        except:
            self.error = []
        self.optimizer = torch.optim.Adam([optim_var], lr=self.lr)
        if isinstance(lambda_tv, type(None)):
            self._lfun_ = self.lossfn
        else:
            self._ltv_ = lambda_tv  # warning: no serious type checking happening here...
            self._lfun_ = self.lossTV
        if isinstance(minibatch_plan, type(None)):  # mini-batch optimization not requested
            minibatch_plan = {
                'minibatches': [1],                                 # 1 epoch
                'minibatch_size': [len(self.bragg_condition)],     # batch optimization
                'iterations_per_minibatch': [default_iterations]
            }
        self.buildCustomPlan(minibatch_plan)
        return

    def buildCustomPlan(self, minibatch_plan):
        N = list(range(len(self.peaks)))
        scheme = [
            [[sz, it]] * num
            for sz, it, num in zip(
                minibatch_plan['minibatch_size'],
                minibatch_plan['iterations_per_minibatch'],
                minibatch_plan['minibatches']
            )
        ]
        self.optimization_plan = [
            [
                [random.sample(N, sch[0]), sch[1]]
                for sch in epoch
            ]
            for epoch in scheme
        ]
        return

    def unitCellClamp(self, tol=1.e-6):
        """
        Clamps distortions to crystallographic unit cell.
        """
        self.__u__ = self.x[self.N:-len(self.bragg_condition)].reshape(3, -1)
        temp = torch.linalg.solve(self.basis_real, self.__u__)
        with torch.no_grad():
            temp[:] = temp.clamp(-0.5+tol, 0.5-tol)  # keep it just within the unit cell
            self.x[self.N:-len(self.bragg_condition)].copy_((self.basis_real @ temp).ravel())
        return

    def lossfn(self):
        return sum(self.losses)

    def penaltyTV(self):
        mag = self.x[:self.N].reshape(*(self.cubeSize for p in range(3)))
        tvmag = TV3D(mag)
        return tvmag

    def lossTV(self):
        return self.lossfn() + self._ltv_*self.penaltyTV()

    def run(self, epochs=3000):
        for epoch in tqdm(
            self.optimization_plan,
            desc='Epoch          ',
            total=len(self.optimization_plan)
        ):
            for batch in tqdm(epoch, desc='Batch/minibatch', total=len(epoch), leave=False):
                for iteration in tqdm(range(batch[1]), desc='Iteration      ', leave=False):
                    self.optimizer.zero_grad()
                    self.losses = []
                    for n in batch[0]:
                        rho_m = self.getObjectInMount(n)
                        frho_m = fftn_t(rho_m, norm='ortho')
                        self.losses.append(
                            torch.mean((torch.abs(frho_m) - self.bragg_condition[n]['data'])**2)
                        )
                    self.loss_total = self._lfun_()
                    self.loss_total.backward()
                    self.optimizer.step()
                    self.error.append(float(self.loss_total.cpu()))
                    self.unitCellClamp()
            self.medianFilter()

        return


#######################################################################
#                    MultiReflectionSolver Class
#######################################################################

class MultiReflectionSolver(
        LatticePlugin,
        ObjectPlugin,
        OptimizerPlugin
    ):
    """
    Multiple reflection BCDI optimization solver.
    
    Combines lattice calculations, object creation, and optimization
    into a unified solver for multi-reflection Bragg Coherent Diffraction Imaging.
    """

    def __init__(self,
                 database,
                 signal_label='signal',
                 size=128, sigma=3.,
                 activation_parameter=0.75,
                 learning_rate=0.01,
                 minibatch_plan=None,
                 lambda_tv=None,
                 medfilter_kernel=3,
                 init_amplitude_state=None,
                 scale_method='photon_max',
                 cuda_device=0):
        self.loadGlobals(database)
        self.getLatticeBases(*tuple(self.globals['lattice_parms']))
        self.getRotatedCrystalBases()
        self.determineDimensions(size, sigma)
        self.createObjectBB = self.createObjectBB_full  # inherited from ObjectPlugin
        self.prepareScans(label=signal_label)
        self.createUnknowns(activation_parameter, init_amplitude_state, scale_method)
        self.initializeOptimizer(
            self.x,
            learning_rate=learning_rate,
            lambda_tv=lambda_tv,
            minibatch_plan=minibatch_plan
        )
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
        self._domainSize = tuple(self.size for n in range(3))
        self.cubeSize = np.round(size / sigma).astype(int)
        self.cubeSize += (self.cubeSize % 2)
        buff = (size-self.cubeSize)//2
        self._buffer = torch.nn.ConstantPad3d(tuple(buff for n in range(6)), 0)
        return

    def createUnknowns(self, activation_parameter, init_amplitude_state, scale_method):
        self.activ_a = activation_parameter
        self.N = self.cubeSize**3
        if isinstance(init_amplitude_state, type(None)):
            mag = 2.*np.ones(self.N)
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
            __amp__ = 0.5*(1. + torch.tanh(init_t / self.activ_a))
            __u__ = torch.from_numpy(np.zeros((3, self.N))).cuda()
            phs = self.peaks[n] @ __u__
            objBB = (__amp__ * torch.exp(2.j * np.pi * phs)).reshape(*(self.cubeSize for n in range(3)))
            self.bragg_condition[n]['mount'].refreshObject(fftshift_t(self._buffer(objBB)))
            self.bragg_condition[n]['mount'].mountObject()
            rho_m = self.bragg_condition[n]['mount'].getMountedObject()
            if method == 'photon_max':
                frho_m = fftshift_t(fftn_t(fftshift_t(rho_m), norm='ortho')).detach().cpu().numpy()
                scl = np.sqrt(self.bragg_condition[n]['data'].detach().cpu().numpy().max() / (np.absolute(frho_m)**2).max())
                norms.append(scl)
            elif method == 'energy':
                norms.append(
                    self.bragg_condition[n]['data'].detach().cpu().numpy().sum() / (np.absolute(rho_m.detach().cpu().numpy())**2).sum()
                )
            else:
                logger.error('Should set either \'photon_max\' or \'energy\' for scale_method.')
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
            self.peaks.append(
                torch.from_numpy(
                    self.database[scan].attrs['peak'][np.newaxis, :] * 10.
                ).cuda()  # convert units from angstrom^-1 to nm^-1
            )
            self.miller_idx.append(self.database[scan].attrs['miller_idx'])
            scan_data = self.database['%s/%s' % (scan, label)][:]
            R = self.database[scan].attrs['RtoBragg']
            Bdet = self.database[scan].attrs['Bdet']
            Br = self.database[scan].attrs['Breal']  # columns are in units of nm
            mnt = MountPlugin(self.size, Bdet.T @ R, Br, self.globals['step_cubic'])
            bcond = {
                'data': torch.from_numpy(fftshift_o(np.sqrt(scan_data))).cuda(),
                'mount': mnt
            }
            self.bragg_condition.append(bcond)
        self.d_spacing = self.getPlaneSeparations()
        self.d_jumps = [mg**2 * pk.T for mg, pk in zip(self.d_spacing, self.peaks)]
        return

    def getObjectInMount(self, n):
        obj = self.createObjectBB(n)
        self.bragg_condition[n]['mount'].refreshObject(fftshift_t(self._buffer(obj)))
        self.bragg_condition[n]['mount'].mountObject()
        rho_m = self.bragg_condition[n]['mount'].getMountedObject()
        return rho_m

    def centerObject(self):
        state = self.x[:self.N].reshape(*(self.cubeSize for n in range(3)))
        amp = 0.5*(1. + torch.tanh(state / self.activ_a)).detach().cpu().numpy()
        u = self.x[self.N:(4*self.N)].reshape(3, -1).detach().cpu().numpy()
        ux, uy, uz = tuple(arr.reshape(*(self.cubeSize for n in range(3))) for arr in u)
        grid = np.mgrid[
            -self.cubeSize//2:self.cubeSize//2,
            -self.cubeSize//2:self.cubeSize//2,
            -self.cubeSize//2:self.cubeSize//2
        ]
        shift = [-np.round((arr*amp).sum() / amp.sum()).astype(int) for arr in grid]
        state_c = np.roll(state.detach().cpu().numpy(), shift, axis=[0, 1, 2])
        ux_c, uy_c, uz_c = tuple(np.roll(arr, shift, axis=[0, 1, 2]) for arr in [ux, uy, uz])
        xi = self.x[-len(self.bragg_condition):].detach().cpu().numpy()
        my_x = np.concatenate(
            tuple(
                arr.ravel()
                for arr in [state_c, ux_c, uy_c, uz_c, xi]
            )
        )
        new_x = torch.from_numpy(my_x).requires_grad_().cuda()
        with torch.no_grad():
            self.x.copy_(new_x)
        return

    # restricted optimization methods begin from here.

    def setUpRestrictedOptimization(self, new_plan, amp_threshold=0.1, lambda_tv=None, learning_rate=None, median_filter=False):
        self.getSupportVars(amp_threshold, median_filter=median_filter)
        self.createObjectBB = self.createObjectBB_part

        # retain old values of optimization parameters if new ones not specified
        if not isinstance(lambda_tv, type(None)):
            self._ltv_ = lambda_tv
        if not isinstance(learning_rate, type(None)):
            self.lr = learning_rate

        self.initializeOptimizer(
            self.x_new,  # now optimizing over only support voxels.
            learning_rate=self.lr,
            lambda_tv=self._ltv_,
            minibatch_plan=new_plan  # should in general be different from old plan
        )
        return

    def getSupportVars(self, amp_threshold, median_filter):
        self.bin = []  # stores calculated values
        if median_filter:
            my_x = self.medianFilter()
        else:
            my_x = self.x.detach().cpu().numpy()
        ln = my_x.size
        self.bin.append(my_x)
        state = my_x[:self.N]
        amp = 0.5*(1. + np.tanh(state / self.activ_a))
        here_c = np.where(amp > amp_threshold)[0]  # these are the support voxels
        here_c = np.concatenate(tuple(n*self.N + here_c for n in range(4)))  # extend for all u's
        here_c = np.concatenate((here_c, np.array([ln-len(self.bragg_condition)+n for n in range(len(self.bragg_condition))])))
        self.bin.append(here_c)
        my_new_x = my_x[here_c]  # only these voxels are optimized from here on.
        self.x_new = torch.from_numpy(my_new_x).cuda().requires_grad_()
        these = np.zeros(my_x.size)
        these[here_c] = 1.
        self.these_only = torch.tensor(these, dtype=torch.bool).cuda()  # indexes of optimized variables
        return

    def medianFilter(self):
        my_x = self.x.detach().cpu().numpy()
        arr = my_x[:-len(self.bragg_condition)]
        scalers = my_x[-len(self.bragg_condition):]
        arr_by4 = arr.reshape(4, -1)
        state, ux, uy, uz = tuple(ar.reshape(*(self.cubeSize for n in range(3))) for ar in arr_by4)
        amp = 0.5*(1. + np.tanh(state / self.activ_a))
        ux, uy, uz = tuple(median_filter(amp*ar, size=self.mfk) for ar in [ux, uy, uz])
        arr_out = np.concatenate(tuple(ar.ravel() for ar in [state, ux, uy, uz, scalers]))
        with torch.no_grad():
            self.x.copy_(torch.from_numpy(arr_out).cuda())
        return arr_out


#######################################################################
#                         Main Script
#######################################################################

if __name__ == "__main__":
    # Get the script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Data file paths (using relative paths from script directory)
    datafile = os.path.join(script_dir, './data/SiC_400nm.h5')
    resultfile = os.path.join(script_dir, './reconstructions/Results_SiC.h5')
    plot_dir = os.path.join(script_dir, './reconstructions')
    
    # Ensure output directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Defining the optimization sequence
    # ============ ORIGINAL HYPERPARAMETERS (~90,480 + 1,000 iterations) ============
    # optim_plan = {
    #     'minibatches': [800, 640, 480, 320, 160, 80, 40, 1],
    #     'minibatch_size': [4, 4, 4, 5, 5, 5, 5, 6],
    #     'iterations_per_minibatch': [6, 12, 25, 50, 100, 200, 400, 2000]
    # }
    # new_plan = {
    #     'minibatches': [1],
    #     'minibatch_size': [6],
    #     'iterations_per_minibatch': [1000]
    # }
    
    # ============ REDUCED HYPERPARAMETERS (~9,048 + 200 iterations, ~10x faster) ============
    # optim_plan = {
    #     'minibatches': [80, 64, 48, 32, 16, 8, 4, 1],      # Reduced by 10x
    #     'minibatch_size': [4, 4, 4, 5, 5, 5, 5, 6],        # Keep same
    #     'iterations_per_minibatch': [6, 12, 25, 50, 100, 200, 400, 500]  # Last stage reduced
    # }
    # new_plan = {
    #     'minibatches': [1],
    #     'minibatch_size': [6],
    #     'iterations_per_minibatch': [200]  # Reduced from 1000
    # }
    
    # ============ ULTRA-FAST SANITY CHECK (~47 iterations, ~1000x faster) ============
    optim_plan = {
        'minibatches': [1, 1, 1, 1, 1, 1, 1, 1],           # Minimal: 1 minibatch per stage
        'minibatch_size': [4, 4, 4, 5, 5, 5, 5, 6],        # Keep same
        'iterations_per_minibatch': [1, 1, 1, 1, 1, 1, 1, 20]  # Minimal iterations
    }
    new_plan = {
        'minibatches': [1],
        'minibatch_size': [6],
        'iterations_per_minibatch': [20]  # Ultra-fast sanity check
    }

    # Load data
    data = h5.File(datafile, 'r')
    logger.info('Printing data tree...')
    data.visit(print)

    data['scans'].attrs['successful_scans']

    logger.info('Loading grain data into solver.')
    solver = MultiReflectionSolver(
        database=data,
        signal_label='signal',
        sigma=4.93,
        learning_rate=2.5e-3,
        activation_parameter=1.,
        minibatch_plan=optim_plan,
        lambda_tv=1.e-5,
        cuda_device=0
    )
    logger.info('Created solver.')
    logger.info('Successful scans: %s' % str(data['scans'].attrs['successful_scans']))

    logger.info('Global parameters: ')
    for key, value in solver.globals.items():
        logger.info('\t%s: %s' % (key, value))

    logger.info('Running optimizer (Phase 1)...')
    solver.run()

    # Save Phase 1 error plot
    fig = plt.figure()
    plt.semilogy(solver.error)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\mathcal{L}(A, u)$')
    plt.grid()
    plt.title('Objective function (Poisson-stabilized) - Phase 1')
    fig.savefig(os.path.join(plot_dir, 'SiC_error_phase1.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved Phase 1 error plot.')

    solver.centerObject()
    solver.setUpRestrictedOptimization(new_plan=new_plan, amp_threshold=0.25, learning_rate=1.e-3)
    logger.info('Running optimizer (Phase 2)...')
    solver.run()

    amp = solver.__amp__.reshape(
        *(solver.cubeSize for n in range(3))
    ).detach().cpu()
    u_recon = solver.__u__.detach().cpu()

    # Save Phase 2 error plot
    fig = plt.figure()
    plt.semilogy(solver.error)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\mathcal{L}(A, u)$')
    plt.grid()
    plt.title('Objective function (Poisson-stabilized) - Phase 2')
    fig.savefig(os.path.join(plot_dir, 'SiC_error_phase2.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved Phase 2 error plot.')

    # Save amplitude plot
    fig, im, ax = TilePlot(
        tuple(np.transpose(amp, np.roll([0, 1, 2], n))[:, :, solver.cubeSize//2] for n in range(3)),
        (1, 3),
        (15, 5)
    )
    fig.suptitle('Reconstructed electron density (normalized)')
    for n, st in enumerate(['XY', 'YZ', 'ZX']):
        ax[n].set_title(st)
    fig.savefig(os.path.join(plot_dir, 'SiC_amplitude.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved amplitude plot.')

    # Save displacement plot
    fig, im, ax = TilePlot(
        tuple(
            np.transpose(arr.reshape(*(solver.cubeSize for m in range(3))), np.roll([0, 1, 2], n))[:, :, solver.cubeSize//2]
            for n in range(3)
            for arr in u_recon
        ),
        (3, 3),
        (15, 12)
    )
    fig.suptitle('Lattice displacement field components')
    fig.savefig(os.path.join(plot_dir, 'SiC_displacement.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved displacement plot.')

    # Save diffraction pattern comparisons
    for n in range(len(solver.bragg_condition)):
        dat = fftshift_o(solver.bragg_condition[n]['data'].detach().cpu().numpy())
        obj = solver.getObjectInMount(n)
        fobj = fftshift_t(fftn_t(obj)).detach().cpu().numpy()
        intens = np.absolute(fobj)**2

        fig, im, ax = TilePlot(
            (dat[:, :, 64], intens[:, :, 64]),
            (1, 2),
            (10, 5),
            log_norm=True
        )
        ax[0].set_title('Measured')
        ax[1].set_title('Reconstructed')
        fig.suptitle('Diffraction pattern comparison - Scan %d' % n)
        fig.savefig(os.path.join(plot_dir, 'SiC_diffraction_%d.png' % n), dpi=150, bbox_inches='tight')
        plt.close(fig)
    logger.info('Saved diffraction pattern plots.')

    data.close()

    # Save results to HDF5
    logger.info('Saving results to %s' % resultfile)
    res = h5.File(resultfile, 'w')

    res.create_group('real_space')
    res['real_space'].create_dataset('electron_density', data=amp)
    res['real_space'].create_dataset('lattice_deformation', data=u_recon)
    res['real_space'].attrs['scale'] = list(
        solver.x[-len(solver.bragg_condition):].detach().cpu().numpy()
    )

    res.create_dataset('reconstruction_error', data=solver.error)

    res.create_group('reciprocal_space')
    res['reciprocal_space'].attrs['scan_names'] = solver.scan_list
    for n, scan in enumerate([st.replace('scans/', '') for st in solver.scan_list]):
        print(n, scan)
        obj = solver.getObjectInMount(n)
        fobj = fftshift_t(fftn_t(obj)).detach().cpu().numpy()
        res['reciprocal_space'].create_group(scan)
        res['reciprocal_space/%s' % scan].create_dataset('wave', data=fobj)
        res['reciprocal_space/%s' % scan].create_dataset('data', data=fftshift_o(solver.bragg_condition[n]['data'].detach().cpu().numpy()))

    res.close()

    logger.info('Results saved to %s' % resultfile)
    logger.info('All plots saved to %s' % plot_dir)
    logger.info('Done!')

