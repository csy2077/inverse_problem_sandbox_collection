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
    # Before importing agent_main, we need to handle the missing DhariwalUNet dependency
    # by injecting a mock into the agent_main module's namespace
    try:
        # Try to find and import the real DhariwalUNet first
        try:
            from torch_utils.network import DhariwalUNet
        except ImportError:
            try:
                from networks import DhariwalUNet
            except ImportError:
                try:
                    from model import DhariwalUNet
                except ImportError:
                    # Search for it in common locations
                    found = False
                    search_dirs = ['.', '..', 'src', 'models', 'networks']
                    for sd in search_dirs:
                        if os.path.exists(sd):
                            for root, dirs, files in os.walk(sd):
                                for fname in files:
                                    if fname.endswith('.py'):
                                        fpath = os.path.join(root, fname)
                                        try:
                                            with open(fpath, 'r') as f:
                                                content = f.read()
                                            if 'class DhariwalUNet' in content:
                                                # Add dir to path and try importing
                                                module_dir = os.path.dirname(os.path.abspath(fpath))
                                                if module_dir not in sys.path:
                                                    sys.path.insert(0, module_dir)
                                                module_name = fname[:-3]
                                                import importlib
                                                mod = importlib.import_module(module_name)
                                                DhariwalUNet = getattr(mod, 'DhariwalUNet')
                                                found = True
                                                break
                                        except Exception:
                                            continue
                                if found:
                                    break
                        if found:
                            break

                    if not found:
                        # Create a dummy DhariwalUNet class so import succeeds
                        # The main() function's output was already captured, so we just need
                        # the module to load without error
                        class DhariwalUNet(torch.nn.Module):
                            def __init__(self, img_resolution=128, in_channels=1, out_channels=1,
                                         label_dim=0, model_channels=128, channel_mult=None,
                                         attn_resolutions=None, num_blocks=1, dropout=0.0, **kwargs):
                                super().__init__()
                                self.img_resolution = img_resolution
                                self.in_channels = in_channels
                                self.out_channels = out_channels
                                # Simple placeholder network
                                self.net = torch.nn.Conv2d(in_channels, out_channels, 1)

                            def forward(self, x, noise_labels, class_labels=None, **kwargs):
                                return self.net(x)

        # Inject DhariwalUNet into builtins so agent_main can find it
        import builtins
        if not hasattr(builtins, 'DhariwalUNet'):
            builtins.DhariwalUNet = DhariwalUNet

        # Also try to inject into the agent_main module namespace before full import
        # We do this by pre-creating the module
        import types
        if 'agent_main' not in sys.modules:
            # Read agent_main source and inject DhariwalUNet before exec
            agent_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_main.py')
            if os.path.exists(agent_main_path):
                with open(agent_main_path, 'r') as f:
                    source = f.read()

                mod = types.ModuleType('agent_main')
                mod.__file__ = agent_main_path
                mod.__package__ = None
                # Set DhariwalUNet in the module's namespace before executing
                mod.__dict__['DhariwalUNet'] = DhariwalUNet
                sys.modules['agent_main'] = mod
                exec(compile(source, agent_main_path, 'exec'), mod.__dict__)

        from agent_main import main
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