import os


# --- Extracted Dependencies ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(*args):
    """Convert relative path to absolute path based on script location."""
    return os.path.join(SCRIPT_DIR, *args)
