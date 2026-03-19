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
def _setup_dhariwal_unet():
    """Create or import DhariwalUNet and inject it into the agent_main module's namespace."""
    import builtins
    
    # Try to import from various possible locations
    DhariwalUNet = None
    
    # Try torch_utils or similar
    try:
        from torch_utils import DhariwalUNet as _DU
        DhariwalUNet = _DU
    except ImportError:
        pass
    
    if DhariwalUNet is None:
        try:
            from guided_diffusion.unet import UNetModel as _DU
            DhariwalUNet = _DU
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from networks import DhariwalUNet as _DU
            DhariwalUNet = _DU
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from model import DhariwalUNet as _DU
            DhariwalUNet = _DU
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            from diffusion_model import DhariwalUNet as _DU
            DhariwalUNet = _DU
        except ImportError:
            pass

    if DhariwalUNet is None:
        try:
            # Search for any module containing DhariwalUNet in the current directory
            import importlib
            import glob
            py_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*.py'))
            for py_file in py_files:
                mod_name = os.path.splitext(os.path.basename(py_file))[0]
                if mod_name in ('test_main', 'agent_main', 'setup'):
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, 'DhariwalUNet'):
                        DhariwalUNet = getattr(mod, 'DhariwalUNet')
                        break
                except:
                    continue
        except:
            pass

    if DhariwalUNet is None:
        # Try loading from dill/pickle files that might contain the class
        try:
            data_path = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_diffpir_sandbox/run_code/std_data/data_main.pkl'
            with open(data_path, 'rb') as f:
                tmp_data = dill.load(f)
            # Check if output contains a model with the class
            output = tmp_data.get('output', None)
            if output is not None and hasattr(output, '__class__'):
                # Try to get the module
                mod = sys.modules.get(output.__class__.__module__, None)
                if mod and hasattr(mod, 'DhariwalUNet'):
                    DhariwalUNet = getattr(mod, 'DhariwalUNet')
        except:
            pass

    if DhariwalUNet is None:
        # Create a stub class that can be used as a placeholder
        # This is needed because agent_main.py references DhariwalUNet at module level
        # We need to define a working UNet architecture
        class DhariwalUNet(torch.nn.Module):
            """Stub/placeholder for DhariwalUNet - implements a minimal UNet."""
            def __init__(self, img_resolution, in_channels, out_channels, label_dim=0,
                         model_channels=128, channel_mult=None, attn_resolutions=None,
                         num_blocks=1, dropout=0.0, **kwargs):
                super().__init__()
                self.img_resolution = img_resolution
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.label_dim = label_dim
                self.model_channels = model_channels
                
                # Simple conv network as placeholder
                ch = model_channels
                self.input_conv = torch.nn.Conv2d(in_channels, ch, 3, padding=1)
                self.time_embed = torch.nn.Sequential(
                    torch.nn.Linear(1, ch),
                    torch.nn.SiLU(),
                    torch.nn.Linear(ch, ch),
                )
                if label_dim > 0:
                    self.label_embed = torch.nn.Linear(label_dim, ch)
                self.blocks = torch.nn.ModuleList()
                for _ in range(num_blocks):
                    self.blocks.append(torch.nn.Sequential(
                        torch.nn.GroupNorm(min(32, ch), ch),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(ch, ch, 3, padding=1),
                        torch.nn.GroupNorm(min(32, ch), ch),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(ch, ch, 3, padding=1),
                    ))
                self.output_conv = torch.nn.Sequential(
                    torch.nn.GroupNorm(min(32, ch), ch),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(ch, out_channels, 3, padding=1),
                )

            def forward(self, x, noise_labels, class_labels=None, **kwargs):
                dtype = x.dtype
                h = self.input_conv(x)
                temb = self.time_embed(noise_labels.reshape(-1, 1).to(dtype))
                h = h + temb.reshape(temb.shape[0], temb.shape[1], 1, 1)
                if class_labels is not None and self.label_dim > 0:
                    h = h + self.label_embed(class_labels).unsqueeze(-1).unsqueeze(-1)
                for block in self.blocks:
                    h = h + block(h)
                out = self.output_conv(h)
                return out.to(dtype)

    # Inject into builtins so it's available everywhere
    builtins.DhariwalUNet = DhariwalUNet
    return DhariwalUNet


# Setup DhariwalUNet before importing agent_main
DhariwalUNet = _setup_dhariwal_unet()

# Now patch it into the agent_main module namespace if needed
# We need to make sure it's available when agent_main.py is parsed
import importlib

# Pre-create the agent_main module entry if it doesn't exist
if 'agent_main' not in sys.modules:
    # Load the source and inject DhariwalUNet
    agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
    if os.path.exists(agent_main_path):
        import types
        
        # Read the source
        with open(agent_main_path, 'r') as f:
            source = f.read()
        
        # Create module
        mod = types.ModuleType('agent_main')
        mod.__file__ = agent_main_path
        mod.__package__ = None
        
        # Add DhariwalUNet to the module's namespace before executing
        mod.__dict__['DhariwalUNet'] = DhariwalUNet
        
        # Also add all standard imports that might be needed
        sys.modules['agent_main'] = mod
        
        try:
            exec(compile(source, agent_main_path, 'exec'), mod.__dict__)
        except Exception as e:
            # Remove failed module
            del sys.modules['agent_main']
            print(f"WARNING: Failed to pre-load agent_main: {e}")
            traceback.print_exc()

from verification_utils import recursive_check


def test_main():
    # Data paths
    data_paths = [
        '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_diffpir_sandbox/run_code/std_data/data_main.pkl'
    ]

    # Classify paths into outer (standard) and inner (parent_function) paths
    outer_path = None
    inner_paths = []

    for p in data_paths:
        basename = os.path.basename(p)
        if 'parent_function' in basename or 'parent_' in basename:
            inner_paths.append(p)
        else:
            outer_path = p

    if outer_path is None:
        print("ERROR: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    # Phase 1: Load outer data
    print(f"Loading outer data from: {outer_path}")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load outer data: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer data loaded. func_name: {outer_data.get('func_name', 'unknown')}")
    print(f"  args count: {len(outer_args)}, kwargs keys: {list(outer_kwargs.keys())}")

    # Determine scenario
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Run main to get the operator
        print("Running main(*args, **kwargs) to get operator...")
        try:
            from agent_main import main as target_main
            agent_operator = target_main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"WARNING: agent_operator is not callable (type={type(agent_operator)}). "
                  f"Attempting to use it as the result directly.")

        # Phase 2: Load inner data and execute
        for inner_path in sorted(inner_paths):
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

            print(f"Inner data loaded. func_name: {inner_data.get('func_name', 'unknown')}")
            print(f"  inner args count: {len(inner_args)}, inner kwargs keys: {list(inner_kwargs.keys())}")

            # Execute the operator
            print("Executing agent_operator(*inner_args, **inner_kwargs)...")
            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"ERROR: Failed to execute agent_operator: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Compare results
            print("Comparing results...")
            try:
                passed, msg = recursive_check(inner_expected, actual_result)
            except Exception as e:
                print(f"ERROR: recursive_check failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"TEST FAILED: {msg}")
                sys.exit(1)
            else:
                print(f"Inner test passed for {os.path.basename(inner_path)}")

        print("\nTEST PASSED")
        sys.exit(0)

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        # Run main
        print("Running main(*args, **kwargs)...")
        try:
            from agent_main import main as target_main
            actual_result = target_main(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"ERROR: Failed to run main(): {e}")
            traceback.print_exc()
            sys.exit(1)

        # Compare results
        print("Comparing results...")
        try:
            passed, msg = recursive_check(expected_output, actual_result)
        except Exception as e:
            print(f"ERROR: recursive_check failed with exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"TEST FAILED: {msg}")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)


if __name__ == '__main__':
    test_main()