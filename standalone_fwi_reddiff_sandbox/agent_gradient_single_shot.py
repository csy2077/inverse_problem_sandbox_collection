import numpy as np
from devito import Function
from examples.seismic import Model, Receiver, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver


# --- Extracted Dependencies ---

class fg_pair:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __add__(self, other):
        f = self.f + other.f
        g = self.g + other.g
        return fg_pair(f, g)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

def gradient_single_shot(geometry, d_obs, model, fs=True):
    grad_devito = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    d_pred, u0 = solver.forward(vp=model.vp, save=True)[0:2]
    residual.data[:] = d_pred.data[:] - d_obs.resample(geometry.dt).data[:][0:d_pred.data.shape[0], :]
    fval = .5*np.linalg.norm(residual.data.flatten())**2
    solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad_devito)
    nbl = model.nbl
    z_start = 0 if fs else nbl
    grad_crop = np.array(grad_devito.data[:])[nbl:-nbl, z_start:-nbl]
    return fg_pair(fval, grad_crop)
