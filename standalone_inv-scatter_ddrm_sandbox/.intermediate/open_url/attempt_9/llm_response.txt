```python
import sys
import os
import dill
import traceback
import numpy as np

# Import the target function
from agent_open_url import open_url

# Import verification utility
from verification_utils import recursive_check

# --- Helper Injection (from gen_data_code) ---
# These are needed to ensure dill.load doesn't fail if the pickled data references them
import re
import io
import glob
import hashlib
import tempfile
from typing import Any, Dict
import urllib
import requests

_cache_dir = 'cache'

def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPRO