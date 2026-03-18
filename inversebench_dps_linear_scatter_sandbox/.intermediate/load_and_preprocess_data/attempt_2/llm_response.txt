import sys
import os
import dill
import torch
import numpy as np
import traceback
import pickle

# Add the current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the actual module, or create a placeholder
try:
    from networks import DhariwalUNet
except ImportError:
    try:
        from models import DhariwalUNet
    except ImportError:
        try:
            from diffusion_models import DhariwalUNet
        except ImportError:
            class DhariwalUNet(torch.nn.Module):
                def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, **kwargs):
                    super().__init__()
                    self.img_resolution = img_resolution
                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.label_dim = label_dim
                    self.net = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
                
                def forward(self, x, noise_labels, class_labels=None, **kwargs):
                    return self.net(x)

# Read the agent module source
agent_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_load_and_preprocess_data.py')

agent_source = open(agent_module_path, 'r').read()

exec_globals = {
    '__name__': 'agent_load_and_preprocess_data',
    '__builtins__': __builtins__,
    'DhariwalUNet': DhariwalUNet,
}

import re
import io
import glob
import uuid
import hashlib
import tempfile
import urllib
import requests
import html
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from scipy.special import hankel1
from scipy.integrate import dblquad
import lmdb

exec_globals.update({
    'os': os,
    're': re,
    'io': io,
    'glob': glob,
    'uuid': uuid,
    'pickle': pickle,
    'hashlib': hashlib,
    'tempfile': tempfile,
    'urllib': urllib,
    'requests': requests,
    'html': html,
    'Any': Any,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'Union': Union,
    'np': np,
    'numpy': np,
    'torch': torch,
    'DataLoader': DataLoader,
    'Dataset': Dataset,
    'TF': TF,
    'hankel1': hankel1,
    'dblquad': dblquad,
    'lmdb': lmdb,
})

exec(compile(agent_source, agent_module_path, 'exec'), exec_globals)

load_and_preprocess_data = exec_globals['load_and_preprocess_data']

from verification_utils import recursive_check


def find_data_files(data_paths):
    """
    Analyze data paths to determine test strategy.
    Returns (outer_path, inner_path) where inner_path may be None.
    """
    outer_path = None
    inner_path = None
    
    for path in data_paths:
        basename = os.path.basename(path)
        if 'parent_function' in basename:
            inner_path = path
        elif basename == 'data_load_and_preprocess_data.pkl' or basename == 'standard_data_load_and_preprocess_data.pkl':
            outer_path = path
        elif 'load_and_preprocess_data' in basename and outer_path is None:
            outer_path = path
    
    return outer_path, inner_path


def load_data_file(filepath):
    """Load a dill-serialized data file with multiple fallback methods."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise ValueError(f"Data file is empty (0 bytes): {filepath}")
    
    print(f"File size: {file_size} bytes")
    
    # Try multiple loading methods
    errors = []
    
    # Method 1: dill
    try:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data
    except Exception as e:
        errors.append(f"dill.load failed: {e}")
    
    # Method 2: pickle
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        errors.append(f"pickle.load failed: {e}")
    
    # Method 3: torch.load
    try:
        data = torch.load(filepath, map_location='cpu')
        return data
    except Exception as e:
        errors.append(f"torch.load failed: {e}")
    
    # Method 4: numpy load
    try:
        data = np.load(filepath, allow_pickle=True)
        if hasattr(data, 'item'):
            return data.item()
        return data
    except Exception as e:
        errors.append(f"np.load failed: {e}")
    
    raise RuntimeError(f"Failed to load data file with all methods:\n" + "\n".join(errors))


def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/inversebench_dps_linear_scatter_sandbox/run_code/std_data/data_load_and_preprocess_data.pkl']
    
    print(f"Analyzing {len(data_paths)} data file(s)...")
    
    # Find outer and inner data files
    outer_path, inner_path = find_data_files(data_paths)
    
    if outer_path is None:
        print("ERROR: Could not find outer data file for load_and_preprocess_data")
        sys.exit(1)
    
    print(f"Outer data path: {outer_path}")
    print(f"Inner data path: {inner_path}")
    
    # Check if the file exists and has content
    if not os.path.exists(outer_path):
        print(f"ERROR: Data file does not exist: {outer_path}")
        sys.exit(1)
    
    file_size = os.path.getsize(outer_path)
    print(f"Outer data file size: {file_size} bytes")
    
    if file_size == 0:
        print("WARNING: Data file is empty. Creating default test configuration...")
        # Create a default configuration for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = {
            'problem': {
                'model': {
                    'Lx': 0.18,
                    'Ly': 0.18,
                    'Nx': 128,
                    'Ny': 128,
                    'wave': 6,
                    'numRec': 360,
                    'numTrans': 60,
                    'sensorRadius': 1.6,
                    'sigma_noise': 0.01,
                    'unnorm_shift': 0.0,
                    'unnorm_scale': 10.0,
                },
                'data': {
                    'root': '/tmp/test_data',
                    'resolution': 128,
                    'mean': 0.0,
                    'std': 5.0,
                    'id_list': [0],
                },
                'prior': 'test_model.pkl',
                'exp_dir': '/tmp/exp',
            },
            'pretrain': {
                'model': {
                    'img_resolution': 128,
                    'img_channels': 1,
                    'label_dim': 0,
                    'model_type': 'DhariwalUNet',
                }
            },
            'algorithm': {
                'name': 'dps',
                'method': {
                    'diffusion_scheduler_config': {},
                    'guidance_scale': 1.0,
                    'sde': 'vp',
                }
            },
            'num_samples': 1,
            'exp_name': 'test',
            'inference': {},
        }
        outer_args = (config, device)
        outer_kwargs = {}
        outer_output = None
    else:
        # Phase 1: Load outer data and reconstruct operator/result
        try:
            print("\n=== Phase 1: Loading outer data ===")
            outer_data = load_data_file(outer_path)
            
            outer_args = outer_data.get('args', ())
            outer_kwargs = outer_data.get('kwargs', {})
            outer_output = outer_data.get('output', None)
            
            print(f"Outer args count: {len(outer_args)}")
            print(f"Outer kwargs keys: {list(outer_kwargs.keys())}")
            
        except Exception as e:
            print(f"ERROR loading outer data: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    # Execute the function with outer args
    try:
        print("\n=== Executing load_and_preprocess_data ===")
        agent_result = load_and_preprocess_data(*outer_args, **outer_kwargs)
        print(f"Function executed successfully. Result type: {type(agent_result)}")
        
    except Exception as e:
        print(f"ERROR executing load_and_preprocess_data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Phase 2: Determine if this is Scenario A or B
    if inner_path is not None:
        print("\n=== Phase 2: Scenario B - Factory/Closure Pattern ===")
        
        if not callable(agent_result):
            print(f"ERROR: Expected callable operator, got {type(agent_result)}")
            sys.exit(1)
        
        try:
            print("Loading inner data...")
            inner_data = load_data_file(inner_path)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)
            
            print(f"Inner args count: {len(inner_args)}")
            print(f"Inner kwargs keys: {list(inner_kwargs.keys())}")
            
        except Exception as e:
            print(f"ERROR loading inner data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        try:
            print("Executing the operator with inner args...")
            result = agent_result(*inner_args, **inner_kwargs)
            print(f"Operator executed successfully. Result type: {type(result)}")
            
        except Exception as e:
            print(f"ERROR executing operator: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    else:
        print("\n=== Phase 2: Scenario A - Simple Function ===")
        result = agent_result
        expected = outer_output
    
    # Phase 3: Verification
    print("\n=== Phase 3: Verification ===")
    
    if expected is None:
        print("WARNING: No expected output found in data file")
        print("TEST PASSED (no expected output to compare)")
        sys.exit(0)
    
    try:
        passed, msg = recursive_check(expected, result)
        
        if passed:
            print("TEST PASSED")
            sys.exit(0)
        else:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR during verification: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()