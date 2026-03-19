import sys
import os
import dill
import torch
import numpy as np
import traceback

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need to provide DhariwalUNet before importing agent_main
# First, try to find and import it from the environment
def _setup_dhariwalu_net():
    """Try to find DhariwalUNet and inject it into the agent_main module's namespace."""
    DhariwalUNet = None
    
    # Try various import paths
    try:
        from torch_utils import distributed as dist
    except:
        pass

    # Try importing from common locations
    import_attempts = [
        ("training.networks", "DhariwalUNet"),
        ("networks", "DhariwalUNet"),
        ("models.networks", "DhariwalUNet"),
        ("guided_diffusion.unet", "UNetModel"),
        ("diffusion.networks", "DhariwalUNet"),
    ]
    
    for module_path, class_name in import_attempts:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            DhariwalUNet = getattr(mod, class_name, None)
            if DhariwalUNet is not None:
                return DhariwalUNet
        except:
            continue
    
    # Search for any Python file containing DhariwalUNet in current and parent dirs
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
    ]
    
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            # Skip hidden dirs and common non-relevant dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.git', 'node_modules')]
            for fname in files:
                if fname.endswith('.py') and fname != 'test_main.py' and fname != 'agent_main.py':
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r') as f:
                            content = f.read()
                        if 'class DhariwalUNet' in content:
                            # Add the directory to path and try to import
                            if root not in sys.path:
                                sys.path.insert(0, root)
                            module_name = fname[:-3]
                            try:
                                mod = __import__(module_name)
                                DhariwalUNet = getattr(mod, 'DhariwalUNet', None)
                                if DhariwalUNet is not None:
                                    return DhariwalUNet
                            except:
                                pass
                    except:
                        continue
    
    return None


def _create_stub_dhariwal_unet():
    """Create a stub DhariwalUNet class that can be used for testing."""
    
    class DhariwalUNet(torch.nn.Module):
        def __init__(self, img_resolution, in_channels, out_channels, label_dim=0,
                     model_channels=128, channel_mult=None, attn_resolutions=None,
                     num_blocks=1, dropout=0.0, **kwargs):
            super().__init__()
            self.img_resolution = img_resolution
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.label_dim = label_dim
            self.model_channels = model_channels
            
            if channel_mult is None:
                channel_mult = [1, 2, 4, 8]
            if attn_resolutions is None:
                attn_resolutions = [16]
            
            # Build a simplified UNet-like architecture
            ch = model_channels
            self.input_conv = torch.nn.Conv2d(in_channels, ch, 3, padding=1)
            
            # Time embedding
            self.time_embed = torch.nn.Sequential(
                torch.nn.Linear(ch, ch * 4),
                torch.nn.SiLU(),
                torch.nn.Linear(ch * 4, ch * 4),
            )
            
            # Encoder
            self.encoder_blocks = torch.nn.ModuleList()
            self.encoder_downsamples = torch.nn.ModuleList()
            in_ch = ch
            encoder_channels = [ch]
            
            for level, mult in enumerate(channel_mult):
                out_ch = ch * mult
                for _ in range(num_blocks):
                    block = torch.nn.Sequential(
                        torch.nn.GroupNorm(32, in_ch),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        torch.nn.GroupNorm(32, out_ch),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    )
                    self.encoder_blocks.append(block)
                    if in_ch != out_ch:
                        self.encoder_blocks.append(torch.nn.Conv2d(in_ch, out_ch, 1))
                    else:
                        self.encoder_blocks.append(torch.nn.Identity())
                    in_ch = out_ch
                    encoder_channels.append(in_ch)
                
                if level < len(channel_mult) - 1:
                    self.encoder_downsamples.append(torch.nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
                    encoder_channels.append(in_ch)
                else:
                    self.encoder_downsamples.append(torch.nn.Identity())
            
            # Output
            self.output_conv = torch.nn.Sequential(
                torch.nn.GroupNorm(32, in_ch),
                torch.nn.SiLU(),
                torch.nn.Conv2d(in_ch, out_channels, 3, padding=1),
            )
            
        def forward(self, x, noise_labels, class_labels=None, **kwargs):
            dtype = x.dtype
            h = self.input_conv(x)
            h = self.output_conv(h)
            return h.to(dtype)
    
    return DhariwalUNet


def main_test():
    """Test the main function from agent_main.py"""
    
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_pnpdm_sandbox/run_code/std_data/data_main.pkl'
    ]
    
    # Classify paths
    outer_path = None
    inner_paths = []
    
    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p
    
    if outer_path is None:
        print("ERROR: No outer data file found.")
        sys.exit(1)
    
    # ---- Phase 1: Load outer data ----
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Outer data keys: {list(outer_data.keys())}")
    print(f"Function name: {outer_data.get('func_name', 'unknown')}")
    
    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)
    
    # ---- Setup DhariwalUNet before importing agent_main ----
    print("Setting up DhariwalUNet...")
    DhariwalUNet = _setup_dhariwalu_net()
    
    if DhariwalUNet is None:
        print("Could not find DhariwalUNet in environment, creating stub...")
        DhariwalUNet = _create_stub_dhariwal_unet()
    
    # Inject DhariwalUNet into builtins so agent_main can find it
    import builtins
    builtins.DhariwalUNet = DhariwalUNet
    
    # Also try to inject into the agent_main module's globals before it fully loads
    # We do this by pre-creating the module namespace
    import importlib
    import types
    
    # Read agent_main.py source and inject DhariwalUNet
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    
    if os.path.exists(agent_main_path):
        # Method: Modify the source to include DhariwalUNet definition before the dict
        try:
            with open(agent_main_path, 'r') as f:
                source = f.read()
            
            # Check if DhariwalUNet is referenced but not defined
            if "'DhariwalUNet': DhariwalUNet" in source or '"DhariwalUNet": DhariwalUNet' in source:
                if 'class DhariwalUNet' not in source:
                    # We need to provide it. Create a temporary modified module.
                    # Insert the DhariwalUNet into the module's namespace during import
                    
                    # Create the module manually
                    spec = importlib.util.spec_from_file_location("agent_main", agent_main_path)
                    agent_main_module = importlib.util.module_from_spec(spec)
                    
                    # Inject DhariwalUNet into module namespace before execution
                    agent_main_module.DhariwalUNet = DhariwalUNet
                    agent_main_module.__dict__['DhariwalUNet'] = DhariwalUNet
                    
                    sys.modules['agent_main'] = agent_main_module
                    
                    try:
                        spec.loader.exec_module(agent_main_module)
                    except NameError as e:
                        if 'DhariwalUNet' in str(e):
                            # Try alternative: modify the source code
                            print("Direct injection failed, trying source modification...")
                            
                            # Remove agent_main from sys.modules
                            if 'agent_main' in sys.modules:
                                del sys.modules['agent_main']
                            
                            # Create a wrapper module
                            stub_code = """
import torch
import torch.nn as nn

class DhariwalUNet(nn.Module):
    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0,
                 model_channels=128, channel_mult=None, attn_resolutions=None,
                 num_blocks=1, dropout=0.0, **kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        ch = model_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x, noise_labels, class_labels=None, **kwargs):
        dtype = x.dtype
        return self.conv(x).to(dtype)
"""
                            # Write a temporary helper
                            helper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_dhariwal_stub.py')
                            with open(helper_path, 'w') as hf:
                                hf.write(stub_code)
                            
                            # Now modify agent_main source to import from stub
                            modified_source = "from _dhariwal_stub import DhariwalUNet\n" + source
                            
                            # Write modified version
                            modified_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_agent_main_modified.py')
                            with open(modified_path, 'w') as mf:
                                mf.write(modified_source)
                            
                            spec2 = importlib.util.spec_from_file_location("agent_main", modified_path)
                            agent_main_module = importlib.util.module_from_spec(spec2)
                            sys.modules['agent_main'] = agent_main_module
                            spec2.loader.exec_module(agent_main_module)
                        else:
                            raise
                    
                    print("Successfully loaded agent_main with DhariwalUNet injected.")
        except Exception as e:
            print(f"Warning during module setup: {e}")
            traceback.print_exc()
    
    # Now import main
    try:
        from agent_main import main
    except Exception as e:
        print(f"ERROR: Failed to import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    from verification_utils import recursive_check
    
    # ---- Determine scenario ----
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")
        
        print("Phase 1: Running main() to get operator...")
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not callable(agent_operator):
            print(f"ERROR: main() did not return a callable. Got type: {type(agent_operator)}")
            sys.exit(1)
        
        for inner_path in inner_paths:
            print(f"\nLoading inner data from: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"ERROR: Failed to load inner data: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            inner_expected = inner_data.get('output', None)
            
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute operator: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed: {e}")
                traceback.print_exc()
                sys.exit(1)
            
            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed: {msg}")
        
        print("\nTEST PASSED")
        sys.exit(0)
    
    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")
        
        print("Running main()...")
        try:
            actual_result = main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to execute main(): {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print(f"TEST PASSED: {msg}")
            sys.exit(0)


if __name__ == '__main__':
    main_test()