

# --- Extracted Dependencies ---

FLOAT_FORMAT = '{: .17e}'

def strfloat(x):
    """
    Converts a float to a string, with maximum precision.
    """
    return FLOAT_FORMAT.format(float(x))
