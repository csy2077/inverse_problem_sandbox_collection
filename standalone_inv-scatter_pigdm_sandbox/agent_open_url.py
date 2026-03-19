import os
import re
import io
import glob
import tempfile
import hashlib
import urllib
import requests
from typing import Any, Dict


# --- Extracted Dependencies ---

_cache_dir = 'cache'

def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, url_md5 + "_file")
        with open(cache_file, "wb") as f:
            f.write(url_data)
        if return_filename:
            return cache_file
    return io.BytesIO(url_data)
