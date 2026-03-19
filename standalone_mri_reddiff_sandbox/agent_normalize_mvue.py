import numpy as np


# --- Extracted Dependencies ---

def normalize_mvue(gen_img, estimated_mvue):
    scaling = np.quantile(np.abs(estimated_mvue), 0.99)
    return gen_img / scaling
