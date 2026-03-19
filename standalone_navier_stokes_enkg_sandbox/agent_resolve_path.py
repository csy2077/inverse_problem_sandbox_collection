import os


# --- Extracted Dependencies ---

def resolve_path(script_dir, path):
    """Resolve a path relative to the script directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(script_dir, path)
