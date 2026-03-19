import numpy as np
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
        if extraOptions is not None and 'FilterType' in list(extraOptions.keys()):
            cfg['FilterType'] = extraOptions['FilterType']
            opts = {key: extraOptions[key] for key in extraOptions if key != 'FilterType'}
        else:
            opts = extraOptions if extraOptions is not None else {}
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
