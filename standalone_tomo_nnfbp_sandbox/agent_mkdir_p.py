import os
import errno


# --- Extracted Dependencies ---

def mkdir_p(path):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
