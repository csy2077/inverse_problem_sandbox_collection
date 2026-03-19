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
