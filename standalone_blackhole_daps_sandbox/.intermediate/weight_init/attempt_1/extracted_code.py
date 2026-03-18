import sys
import os
import dill
import torch
import numpy as np
import traceback
from agent_weight_init import weight_init
from verification_utils import recursive_check

def main():
    # Data paths provided
    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_blackhole_daps_sandbox/run_code/std_data/data_weight_init.pkl']
    
    # File Logic Setup: Identify outer and inner paths
    outer_path = None
    inner_paths = []
    
    for path in data_paths:
        basename = os.path.basename(path)
        if basename == 'data_weight_init.pkl':
            outer_path = path
        elif 'parent_function' in basename and 'weight_init' in basename:
            inner_paths.append(path)
    
    if not outer_path:
        print("ERROR: No outer data file found (data_weight_init.pkl)")
        sys.exit(1)
    
    # Phase 1: Load outer data and reconstruct operator
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        
        outer_args = outer_data.get('args', ())
        outer_kwargs = outer_data.get('kwargs', {})
        expected = outer_data.get('output')
        
        print(f"[Phase 1] Loading outer data from: {outer_path}")
        print(f"[Phase 1] Outer args: {len(outer_args)} items")
        print(f"[Phase 1] Outer kwargs: {list(outer_kwargs.keys())}")
        
        # Set random seeds for reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Execute weight_init with outer data
        result = weight_init(*outer_args, **outer_kwargs)
        
        print(f"[Phase 1] Result type: {type(result)}")
        print(f"[Phase 1] Expected type: {type(expected)}")
        
        # Verify result
        passed, msg = recursive_check(expected, result)
        
        if not passed:
            print("TEST FAILED")
            print(f"Verification message: {msg}")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("TEST PASSED")
        print("="*50)
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR in Phase 1: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()