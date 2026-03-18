from examples.seismic.acoustic import AcousticWaveSolver


# --- Extracted Dependencies ---

def forward_single_shot(geometry, model, save=False, dt=1.0):
    solver_i = AcousticWaveSolver(model, geometry, space_order=4)
    d_obs = solver_i.forward(vp=model.vp, save=save, dt=dt)[0]
    return d_obs.resample(dt)
