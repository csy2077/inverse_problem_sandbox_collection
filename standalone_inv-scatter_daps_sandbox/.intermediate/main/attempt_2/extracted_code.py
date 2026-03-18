import sys
import os
import dill
import torch
import numpy as np
import traceback

# Determine data paths and classify them
data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_inv-scatter_daps_sandbox/run_code/std_data/data_main.pkl']

outer_path = None
inner_paths = []

for p in data_paths:
    basename = os.path.basename(p)
    if 'parent_function' in basename:
        inner_paths.append(p)
    else:
        outer_path = p


def test_main():
    try:
        # First, find and handle DhariwalUNet
        DhariwalUNet = None
        try:
            from torch_utils.network import DhariwalUNet
        except ImportError:
            pass

        if DhariwalUNet is None:
            try:
                from networks import DhariwalUNet
            except ImportError:
                pass

        if DhariwalUNet is None:
            try:
                from model import DhariwalUNet
            except ImportError:
                pass

        if DhariwalUNet is None:
            # Search for it in common locations
            search_dirs = ['.', '..', 'src', 'models', 'networks']
            for sd in search_dirs:
                if os.path.exists(sd):
                    for root, dirs, files in os.walk(sd):
                        for fname in files:
                            if fname.endswith('.py') and fname != 'test_main.py' and fname != 'agent_main.py':
                                fpath = os.path.join(root, fname)
                                try:
                                    with open(fpath, 'r') as f:
                                        content = f.read()
                                    if 'class DhariwalUNet' in content:
                                        module_dir = os.path.dirname(os.path.abspath(fpath))
                                        if module_dir not in sys.path:
                                            sys.path.insert(0, module_dir)
                                        module_name = fname[:-3]
                                        import importlib
                                        mod = importlib.import_module(module_name)
                                        DhariwalUNet = getattr(mod, 'DhariwalUNet')
                                        break
                                except Exception:
                                    continue
                        if DhariwalUNet is not None:
                            break
                if DhariwalUNet is not None:
                    break

        if DhariwalUNet is None:
            class DhariwalUNet(torch.nn.Module):
                def __init__(self, img_resolution=128, in_channels=1, out_channels=1,
                             label_dim=0, model_channels=128, channel_mult=None,
                             attn_resolutions=None, num_blocks=1, dropout=0.0, **kwargs):
                    super().__init__()
                    self.img_resolution = img_resolution
                    self.in_channels = in_channels
                    self.out_channels = out_channels
                    self.net = torch.nn.Conv2d(in_channels, out_channels, 1)

                def forward(self, x, noise_labels, class_labels=None, **kwargs):
                    return self.net(x)

        # Now read agent_main.py source and inject missing names
        import types

        agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
        if not os.path.exists(agent_main_path):
            print(f"FAIL: agent_main.py not found at {agent_main_path}")
            sys.exit(1)

        with open(agent_main_path, 'r') as f:
            source = f.read()

        # Remove agent_main from sys.modules if already loaded (to force re-exec)
        if 'agent_main' in sys.modules:
            del sys.modules['agent_main']

        mod = types.ModuleType('agent_main')
        mod.__file__ = agent_main_path
        mod.__package__ = None

        # Inject DhariwalUNet
        mod.__dict__['DhariwalUNet'] = DhariwalUNet

        # Inject PIQ_AVAILABLE and related
        mod.__dict__['PIQ_AVAILABLE'] = False
        try:
            import piq
            mod.__dict__['PIQ_AVAILABLE'] = True
            mod.__dict__['piq'] = piq
            mod.__dict__['psnr'] = piq.psnr
            mod.__dict__['ssim'] = piq.ssim
        except ImportError:
            mod.__dict__['PIQ_AVAILABLE'] = False

        sys.modules['agent_main'] = mod
        exec(compile(source, agent_main_path, 'exec'), mod.__dict__)

        main = mod.main
        print("Successfully imported main from agent_main")
    except Exception as e:
        print(f"FAIL: Could not import main from agent_main: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        from verification_utils import recursive_check
    except Exception as e:
        print(f"FAIL: Could not import recursive_check from verification_utils: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Phase 1: Load outer data
    if outer_path is None:
        print("FAIL: No outer data file (data_main.pkl) found.")
        sys.exit(1)

    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print(f"Loaded outer data from: {outer_path}")
        print(f"Outer data keys: {list(outer_data.keys())}")
    except Exception as e:
        print(f"FAIL: Could not load outer data from {outer_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    outer_output = outer_data.get('output', None)

    # Scenario determination
    if len(inner_paths) > 0:
        # Scenario B: Factory/Closure pattern
        print("Detected Scenario B: Factory/Closure pattern")

        # Phase 1: Create operator
        try:
            agent_operator = main(*outer_args, **outer_kwargs)
            print(f"main() returned: {type(agent_operator)}")
        except Exception as e:
            print(f"FAIL: main(*args, **kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: main() did not return a callable. Got {type(agent_operator)}")
            sys.exit(1)

        # Phase 2: Execute with inner data
        for inner_path in sorted(inner_paths):
            print(f"\nProcessing inner data: {inner_path}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
                print(f"Loaded inner data. Keys: {list(inner_data.keys())}")
            except Exception as e:
                print(f"FAIL: Could not load inner data from {inner_path}: {e}")
                traceback.print_exc()
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                actual_result = agent_operator(*inner_args, **inner_kwargs)
                print(f"agent_operator() returned: {type(actual_result)}")
            except Exception as e:
                print(f"FAIL: agent_operator(*inner_args, **inner_kwargs) raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            # Comparison
            try:
                passed, msg = recursive_check(expected, actual_result)
            except Exception as e:
                print(f"FAIL: recursive_check raised an exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {inner_path}")
                print(f"Message: {msg}")
                sys.exit(1)
            else:
                print(f"PASSED for inner data: {os.path.basename(inner_path)}")

    else:
        # Scenario A: Simple function call
        print("Detected Scenario A: Simple function call")

        try:
            actual_result = main(*outer_args, **outer_kwargs)
            print(f"main() returned: {type(actual_result)}")
        except Exception as e:
            print(f"FAIL: main(*args, **kwargs) raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        expected = outer_output

        # Comparison
        try:
            passed, msg = recursive_check(expected, actual_result)
        except Exception as e:
            print(f"FAIL: recursive_check raised an exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            sys.exit(1)
        else:
            print("PASSED for outer data.")

    print("\nTEST PASSED")
    sys.exit(0)


if __name__ == '__main__':
    test_main()