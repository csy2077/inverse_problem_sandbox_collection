from read_roi import read_roi_file


# --- Extracted Dependencies ---

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
