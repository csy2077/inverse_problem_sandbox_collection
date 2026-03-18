import os
import re
import io
import glob
import uuid
import pickle
import hashlib
import tempfile
import urllib
import requests
import html
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from scipy.special import hankel1
from scipy.integrate import dblquad
import lmdb

_cache_dir = 'cache'

def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    url_name = None
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

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) 
        if return_filename:
            return cache_file

    assert not return_filename
    return io.BytesIO(url_data)

def parse_int_list(s):
    if isinstance(s, list): return s
    if isinstance(s, int): return [s]
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

_model_dict = {
    'DhariwalUNet': DhariwalUNet,
}

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        sigma_min       = 0,
        sigma_max       = float('inf'),
        sigma_data      = 0.5,
        model_type      = 'DhariwalUNet',
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        if model_type not in _model_dict:
            raise ValueError(f"Model type '{model_type}' not found in _model_dict. Available: {list(_model_dict.keys())}")
            
        self.model = _model_dict[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

def generate_em_functions(p):
    XPix, YPix = np.meshgrid(p['x'], p['y'])
    hank_fun = lambda x: 1j * 0.25 * hankel1(0, x)
    transmitter_angles = np.linspace(0, 359, p['numTrans']) * np.pi / 180
    x_transmit = p['sensorRadius'] * np.cos(transmitter_angles)
    y_transmit = p['sensorRadius'] * np.sin(transmitter_angles)
    receiver_angles = np.linspace(0, 359, p['numRec']) * np.pi / 180
    x_receive = p['sensorRadius'] * np.cos(receiver_angles)
    y_receive = p['sensorRadius'] * np.sin(receiver_angles)
    p['receiverMask'] = np.ones((p['numTrans'], p['numRec']))
    
    diff_x_rp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(x_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_rp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numRec'])) - \
                np.tile(y_receive[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_rec_to_pix = np.sqrt(diff_x_rp**2 + diff_y_rp**2)

    diff_x_tp = np.tile(XPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(x_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    diff_y_tp = np.tile(YPix[:, :, np.newaxis], (1, 1, p['numTrans'])) - \
                np.tile(y_transmit[np.newaxis, np.newaxis, :], (p['Ny'], p['Nx'], 1))
    distance_trans_to_pix = np.sqrt(diff_x_tp**2 + diff_y_tp**2)
    
    p['uincDom'] = hank_fun(p['kb'] * distance_trans_to_pix)
    sensor_greens_function = hank_fun(p['kb'] * distance_rec_to_pix)
    p['sensorGreensFunction'] = (p['kb']**2) * sensor_greens_function
    
    x_green = np.arange(-p['Nx'], p['Nx']) * p['dx']                        
    y_green = np.arange(-p['Ny'], p['Ny']) * p['dy']
    XGreen, YGreen = np.meshgrid(x_green, y_green)
    R = np.sqrt(XGreen**2 + YGreen**2)
    domain_greens_function = hank_fun(p['kb'] * R)
    
    def integrand_real(x, y):
            if x == 0 and y == 0:
                return 0.0
            return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).real)
    
    def integrand_imag(x, y):
        if x == 0 and y == 0:
            return 0.0
        return np.abs(hank_fun(p['kb'] * np.sqrt(x**2 + y**2)).imag)
    
    Ny = p['Ny']
    Nx = p['Nx']
    dx = p['dx']
    dy = p['dy']
    
    real_part = dblquad(
            integrand_real,
            -dx/2, dx/2, -dy/2, dy/2
        )[0] / (dx * dy)
        
    imag_part = dblquad(
        integrand_imag,
        -dx/2, dx/2, -dy/2, dy/2
    )[0] / (dx * dy)
    
    domain_greens_function[Ny, Nx] = real_part + 1j * imag_part
    
    p['domainGreensFunction'] = (p['kb']**2) * domain_greens_function
    return p

def construct_parameters(Lx=0.18, Ly=0.18, Nx=128, Ny=128, wave=6, numRec=360, numTrans=60, sensorRadius=1.6,
                         device='cuda'):
    em = {}
    em['Lx'] = Lx
    em['Ly'] = Ly
    em['Nx'] = Nx
    em['Ny'] = Ny
    em['dx'] = em['Lx'] / em['Nx']
    em['dy'] = em['Ly'] / em['Ny']
    em['x'] = np.linspace(-em['Nx']/2, em['Nx']/2 - 1, em['Nx']) * em['dx']
    em['y'] = np.linspace(-em['Ny']/2, em['Ny']/2 - 1, em['Ny']) * em['dy']
    em['c'] = 299792458
    em['lambda'] = em['dx'] * wave
    em['freq'] = em['c'] / em['lambda'] / 1e9
    em['numRec'] = numRec
    em['numTrans'] = numTrans
    em['sensorRadius'] = sensorRadius
    em['kb'] = 2 * np.pi / em['lambda']
    em = generate_em_functions(em)
    return torch.from_numpy(em['domainGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['sensorGreensFunction']).to(device).unsqueeze(-1), torch.from_numpy(em['uincDom']).to(device).unsqueeze(-1), torch.from_numpy(em['receiverMask']).unsqueeze(-1)

class LMDBData(Dataset):
    def __init__(self, root, 
                 resolution=128,
                 raw_resolution=128,
                 num_channels=1,
                 norm=True,
                 mean=0.0, std=5.0, id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.raw_resolution = raw_resolution
        self.num_channels = num_channels
        self.norm = norm
        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.idx_map(idx)
        key = f'{idx}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(self.num_channels, self.raw_resolution, self.raw_resolution)
        if self.resolution != self.raw_resolution:
            img = TF.resize(torch.from_numpy(img.copy()), self.resolution, antialias=True)
        if self.norm:
            img = self.normalize(img)
        return {'target': img}

    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def normalize(self, data):
        return (data - self.mean) / (2 * self.std)

    def unnormalize(self, data):
        return data * 2 * self.std + self.mean

def load_and_preprocess_data(config: dict, device: torch.device) -> dict:
    model_config = config['problem']['model']
    Lx = model_config['Lx']
    Ly = model_config['Ly']
    Nx = model_config['Nx']
    Ny = model_config['Ny']
    wave = model_config['wave']
    numRec = model_config['numRec']
    numTrans = model_config['numTrans']
    sensorRadius = model_config['sensorRadius']
    sigma_noise = model_config['sigma_noise']
    unnorm_shift = model_config['unnorm_shift']
    unnorm_scale = model_config['unnorm_scale']
    
    dx = Lx / Nx
    dy = Ly / Ny
    
    domain_greens_function_set, sensor_greens_function_set, uinc_dom_set, receiver_mask_set = \
        construct_parameters(Lx, Ly, Nx, Ny, wave, numRec, numTrans, sensorRadius, device)
    
    sensor_greens_function_set = sensor_greens_function_set.to(torch.complex128)
    uinc_dom_set = uinc_dom_set.to(torch.complex128)
    
    forward_op_params = {
        'dx': dx,
        'dy': dy,
        'Nx': Nx,
        'Ny': Ny,
        'numRec': numRec,
        'numTrans': numTrans,
        'sigma_noise': sigma_noise,
        'unnorm_shift': unnorm_shift,
        'unnorm_scale': unnorm_scale,
        'sensor_greens_function_set': sensor_greens_function_set,
        'uinc_dom_set': uinc_dom_set,
        'device': device
    }
    
    data_config = config['problem']['data']
    testset = LMDBData(
        root=data_config['root'],
        resolution=data_config['resolution'],
        mean=data_config['mean'],
        std=data_config['std'],
        id_list=data_config['id_list']
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    ckpt_path = config['problem']['prior']
    pretrain_config = config['pretrain']['model']
    
    if not os.path.exists(ckpt_path) and not is_url(ckpt_path):
        if os.path.exists(os.path.join('InverseBench', ckpt_path)):
            ckpt_path = os.path.join('InverseBench', ckpt_path)
    
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except:
        net = EDMPrecond(**pretrain_config)
        if is_url(ckpt_path):
            ckpt_file = open_url(ckpt_path, return_filename=True)
            ckpt = torch.load(ckpt_file, map_location=device)
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
        
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        elif 'net' in ckpt.keys():
            net.load_state_dict(ckpt['net'])
        else:
            net.load_state_dict(ckpt)
        net = net.to(device)
    
    net.eval()
    
    scheduler_config = config['algorithm']['method']['diffusion_scheduler_config']
    guidance_scale = config['algorithm']['method']['guidance_scale']
    sde = config['algorithm']['method']['sde']
    
    return {
        'testloader': testloader,
        'testset': testset,
        'net': net,
        'forward_op_params': forward_op_params,
        'scheduler_config': scheduler_config,
        'guidance_scale': guidance_scale,
        'sde': sde,
        'num_samples': config['num_samples'],
        'device': device,
        'exp_dir': config['problem']['exp_dir'],
        'exp_name': config['exp_name'],
        'algorithm_name': config['algorithm']['name'],
        'inference': config['inference']
    }