import numexpr


# --- Extracted Dependencies ---

def sigmoid(x):
    """Sigmoid activation function."""
    return numexpr.evaluate("1./(1.+exp(-x))")
