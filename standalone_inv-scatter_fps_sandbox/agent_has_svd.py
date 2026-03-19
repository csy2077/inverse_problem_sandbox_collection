

# --- Extracted Dependencies ---

def has_svd(forward_op):
    if hasattr(forward_op, 'U') and hasattr(forward_op, 'S') and hasattr(forward_op, 'Vt'):
        return True
    else:
        return False
