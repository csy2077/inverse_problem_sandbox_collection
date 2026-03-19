import os
import numpy as np
import tifffile


# --- Extracted Dependencies ---

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
