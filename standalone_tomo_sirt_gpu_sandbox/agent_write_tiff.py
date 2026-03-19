import tifffile


# --- Extracted Dependencies ---

def write_tiff(fname, img, overwrite=True):
    """Write an array to a TIFF file."""
    if not (fname.endswith('.tif') or fname.endswith('.tiff')):
        fname = fname + '.tiff'
    tifffile.imsave(fname, img)
    print(f"File saved: {fname}")
