import numpy as np


# --- Extracted Dependencies ---

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
