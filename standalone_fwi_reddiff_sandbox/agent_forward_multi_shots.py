from functools import partial
import numpy as np
from examples.seismic.acoustic import AcousticWaveSolver


# --- Extracted Dependencies ---

def convert2np(rec):
    return np.array(rec.data)

def forward_single_shot(geometry, model, save=False, dt=1.0):
    solver_i = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs = solver_i.forward(vp=model.vp, save=save, dt=dt)[0]
    return d_obs.resample(dt)

def forward_multi_shots(model, geometry_list, client, save=False, dt=1.0, return_rec=True):
    forward_single_shot_fn = partial(forward_single_shot, model=model, save=save, dt=dt)
    futures = client.map(forward_single_shot_fn, geometry_list)
    if return_rec:
        shots = client.gather(futures)
        return shots
    else:
        shots_tmp = client.map(convert2np, futures)
        shots_np = client.gather(shots_tmp)
        return shots_np
