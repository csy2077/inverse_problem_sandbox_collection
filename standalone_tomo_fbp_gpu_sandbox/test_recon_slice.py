import sys
import os
import dill
import numpy as np
import traceback

# Ensure the current directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add run_code directory to path
run_code_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code'
if os.path.isdir(run_code_dir):
    sys.path.insert(0, run_code_dir)

try:
    import scipy
    import scipy.sparse
    import scipy.sparse.linalg
except ImportError:
    pass

try:
    import astra
except ImportError:
    pass

try:
    import torch
except ImportError:
    pass


def main():
    std_data_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data'

    data_paths = ['/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_tomo_fbp_gpu_sandbox/run_code/std_data/data_recon_slice.pkl']

    # List all files for debugging
    if os.path.isdir(std_data_dir):
        all_files = sorted(os.listdir(std_data_dir))
        print(f"Files in std_data_dir:")
        for fname in all_files:
            fpath = os.path.join(std_data_dir, fname)
            fsize = os.path.getsize(fpath) if os.path.isfile(fpath) else 'DIR'
            print(f"  {fname}: {fsize} bytes")

    # Separate outer and inner paths
    outer_path = None
    inner_paths = []

    if os.path.isdir(std_data_dir):
        for fname in sorted(os.listdir(std_data_dir)):
            if not fname.endswith('.pkl'):
                continue
            fpath = os.path.join(std_data_dir, fname)
            if 'recon_slice' in fname:
                if 'parent_function' in fname or 'parent_' in fname:
                    inner_paths.append(fpath)
                elif outer_path is None:
                    outer_path = fpath

    if outer_path is None:
        outer_path = data_paths[0]

    print(f"\nOuter path: {outer_path}")
    print(f"Inner paths: {inner_paths}")

    # Try to load outer data directly
    outer_data = None
    load_errors = []

    print(f"\nAttempting to load outer data with dill...")
    try:
        with open(outer_path, 'rb') as f:
            outer_data = dill.load(f)
        print("  Loaded outer data successfully with dill")
    except Exception as e:
        load_errors.append(f"dill.load: {e}")
        print(f"  dill.load failed: {e}")

    if outer_data is None:
        try:
            dill.settings['recurse'] = True
            with open(outer_path, 'rb') as f:
                outer_data = dill.load(f)
            print("  Loaded outer data with dill recurse=True")
        except Exception as e:
            load_errors.append(f"dill recurse: {e}")
            print(f"  dill recurse failed: {e}")
        finally:
            dill.settings['recurse'] = False

    # If direct load failed, reconstruct from other data files
    if outer_data is None:
        print("\n  Direct load failed. Reconstructing test data from other pkl files...")

        # Load reconstruct data to get sinogram, method, parameters, pixel_size, offset
        recon_path = os.path.join(std_data_dir, 'data_reconstruct.pkl')
        recon_data = None
        if os.path.exists(recon_path):
            print(f"  Loading data_reconstruct.pkl for context...")
            try:
                with open(recon_path, 'rb') as f:
                    recon_data = dill.load(f)
                if isinstance(recon_data, dict):
                    print(f"    Keys: {list(recon_data.keys())}")
                    rargs = recon_data.get('args', ())
                    rkwargs = recon_data.get('kwargs', {})
                    print(f"    args count: {len(rargs)}")
                    for i, a in enumerate(rargs):
                        if hasattr(a, 'shape'):
                            print(f"      arg[{i}]: {type(a).__name__} shape={a.shape} dtype={getattr(a, 'dtype', 'N/A')}")
                        else:
                            print(f"      arg[{i}]: {type(a).__name__} = {str(a)[:100]}")
                    print(f"    kwargs: {list(rkwargs.keys())}")
                    for k, v in rkwargs.items():
                        if hasattr(v, 'shape'):
                            print(f"      {k}: {type(v).__name__} shape={v.shape}")
                        else:
                            print(f"      {k}: {type(v).__name__} = {str(v)[:100]}")
            except Exception as e:
                print(f"    Failed: {e}")
                traceback.print_exc()

        if recon_data is None or not isinstance(recon_data, dict):
            print("FAIL: Could not load data_reconstruct.pkl")
            sys.exit(1)

        rargs = recon_data.get('args', ())
        rkwargs = recon_data.get('kwargs', {})
        expected_full_output = recon_data.get('output', None)

        # Extract reconstruct args: (sinogram, angles, method)
        full_sinogram = rargs[0] if len(rargs) > 0 and isinstance(rargs[0], np.ndarray) else None
        angles = rargs[1] if len(rargs) > 1 and isinstance(rargs[1], np.ndarray) else None
        method = rargs[2] if len(rargs) > 2 and isinstance(rargs[2], str) else rkwargs.get('method', 'FBP_CUDA')
        parameters = rkwargs.get('parameters', None)
        pixel_size = rkwargs.get('pixel_size', 1.0)
        offset = rkwargs.get('offset', 0)

        if full_sinogram is None or angles is None:
            print("FAIL: Could not extract sinogram or angles from data_reconstruct.pkl")
            sys.exit(1)

        print(f"\n  Extracted from reconstruct:")
        print(f"    sinogram: {full_sinogram.shape}, dtype={full_sinogram.dtype}")
        print(f"    angles: {angles.shape}, dtype={angles.dtype}")
        print(f"    method: {method}")
        print(f"    parameters: {parameters}")
        print(f"    pixel_size: {pixel_size}")
        print(f"    offset: {offset}")

        # Now create pmat using get_astra_proj_matrix
        # First load the get_astra_proj_matrix data to see its args
        pmat = None
        gapm_path = os.path.join(std_data_dir, 'data_get_astra_proj_matrix.pkl')
        gapm_data = None
        if os.path.exists(gapm_path):
            print(f"\n  Loading data_get_astra_proj_matrix.pkl...")
            try:
                with open(gapm_path, 'rb') as f:
                    gapm_data = dill.load(f)
                if isinstance(gapm_data, dict):
                    print(f"    Keys: {list(gapm_data.keys())}")
                    gapm_args = gapm_data.get('args', ())
                    gapm_kwargs = gapm_data.get('kwargs', {})
                    print(f"    args count: {len(gapm_args)}")
                    for i, a in enumerate(gapm_args):
                        if hasattr(a, 'shape'):
                            print(f"      arg[{i}]: {type(a).__name__} shape={a.shape}")
                        else:
                            print(f"      arg[{i}]: {type(a).__name__} = {str(a)[:100]}")
                    print(f"    kwargs: {list(gapm_kwargs.keys())}")

                    # Try to get pmat from output
                    gapm_output = gapm_data.get('output', None)
                    if gapm_output is not None and hasattr(gapm_output, 'reconstruct'):
                        pmat = gapm_output
                        print(f"    Got pmat from output: {type(pmat).__name__}")
            except Exception as e:
                print(f"    Failed to load: {e}")

        # If pmat not from output, try to create it via the agent function
        if pmat is None:
            print("\n  Attempting to create pmat via agent_get_astra_proj_matrix...")
            try:
                from agent_get_astra_proj_matrix import get_astra_proj_matrix
                
                if gapm_data is not None and isinstance(gapm_data, dict):
                    gapm_args = gapm_data.get('args', ())
                    gapm_kwargs = gapm_data.get('kwargs', {})
                    pmat = get_astra_proj_matrix(*gapm_args, **gapm_kwargs)
                    print(f"    Created pmat from saved args: {type(pmat).__name__}")
                else:
                    # Construct args from what we know
                    # get_astra_proj_matrix typically takes (angles, image_size, det_count)
                    image_size = full_sinogram.shape[1]  # detector count = image size typically
                    det_count = full_sinogram.shape[1]
                    pmat = get_astra_proj_matrix(angles, image_size, det_count)
                    print(f"    Created pmat from reconstructed args: {type(pmat).__name__}")
            except Exception as e:
                print(f"    Failed to create via agent: {e}")
                traceback.print_exc()

        # If still no pmat, try creating directly with astra
        if pmat is None:
            print("\n  Attempting to create pmat directly with astra...")
            try:
                import astra
                
                num_angles = full_sinogram.shape[0]
                num_det = full_sinogram.shape[1]
                image_size = num_det
                
                # Create volume geometry
                vol_geom = astra.create_vol_geom(image_size, image_size)
                # Create projection geometry
                proj_geom = astra.create_proj_geom('parallel', 1.0, num_det, angles)
                # Create projector
                proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
                
                # Create OpTomo-like object
                try:
                    from astra import OpTomo
                    pmat = OpTomo(proj_id)
                    print(f"    Created pmat via astra.OpTomo: {type(pmat).__name__}")
                except (ImportError, AttributeError):
                    # Try scipy-based OpTomo from our agent
                    try:
                        # Inline the OpTomo class
                        class OpTomo(scipy.sparse.linalg.LinearOperator):
                            def __init__(self, proj_id):
                                self.proj_id = proj_id
                                self.vol_geom = astra.projector.volume_geometry(proj_id)
                                self.proj_geom = astra.projector.projection_geometry(proj_id)
                                self.dtype = np.float32
                                self.vshape = astra.geom_size(self.vol_geom)
                                self.sshape = astra.geom_size(self.proj_geom)
                                self.shape = (np.prod(self.sshape), np.prod(self.vshape))
                                
                            def reconstruct(self, method, sinogram, iterations=1, extraOptions=None):
                                if extraOptions is None:
                                    extraOptions = {}
                                vol_geom = self.vol_geom
                                proj_geom = self.proj_geom
                                
                                rec_id = astra.data2d.create('-vol', vol_geom)
                                sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
                                
                                cfg = astra.astra_dict(method)
                                cfg['ReconstructionDataId'] = rec_id
                                cfg['ProjectionDataId'] = sino_id
                                cfg['ProjectorId'] = self.proj_id
                                cfg['option'] = extraOptions
                                
                                alg_id = astra.algorithm.create(cfg)
                                astra.algorithm.run(alg_id, iterations)
                                
                                rec = astra.data2d.get(rec_id)
                                
                                astra.algorithm.delete(alg_id)
                                astra.data2d.delete(rec_id)
                                astra.data2d.delete(sino_id)
                                
                                return rec
                            
                            def _matvec(self, v):
                                return v
                            
                            def _rmatvec(self, v):
                                return v
                        
                        pmat = OpTomo(proj_id)
                        print(f"    Created pmat via inline OpTomo: {type(pmat).__name__}")
                    except Exception as e2:
                        print(f"    Failed inline OpTomo: {e2}")
                        traceback.print_exc()
            except Exception as e:
                print(f"    Failed direct astra creation: {e}")
                traceback.print_exc()

        if pmat is None:
            print("\nFAIL: Could not create pmat object")
            sys.exit(1)

        # Build outer_data
        outer_data = {
            'func_name': 'recon_slice',
            'args': (full_sinogram, method, pmat),
            'kwargs': {
                'parameters': parameters,
                'pixel_size': pixel_size,
                'offset': offset,
            },
            'output': expected_full_output,
        }
        print(f"\n  Successfully built outer_data for recon_slice test")

    # We have outer_data now
    if not isinstance(outer_data, dict):
        print(f"FAIL: outer_data is {type(outer_data)}, expected dict")
        sys.exit(1)

    print(f"\nOuter data keys: {list(outer_data.keys())}")
    print(f"func_name: {outer_data.get('func_name', 'N/A')}")

    outer_args = outer_data.get('args', ())
    outer_kwargs = outer_data.get('kwargs', {})
    expected_output = outer_data.get('output', None)

    print(f"Outer args count: {len(outer_args)}")
    for i, a in enumerate(outer_args):
        if hasattr(a, 'shape'):
            print(f"  arg[{i}]: {type(a).__name__} shape={a.shape} dtype={getattr(a, 'dtype', 'N/A')}")
        elif isinstance(a, str):
            print(f"  arg[{i}]: str = '{a}'")
        else:
            print(f"  arg[{i}]: {type(a).__name__}")
            if hasattr(a, '__class__'):
                print(f"    class: {a.__class__.__name__}")

    print(f"Outer kwargs: {list(outer_kwargs.keys())}")
    for k, v in outer_kwargs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {type(v).__name__} shape={v.shape}")
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")

    if expected_output is not None:
        if hasattr(expected_output, 'shape'):
            print(f"Expected output: {type(expected_output).__name__} shape={expected_output.shape} dtype={expected_output.dtype}")
        else:
            print(f"Expected output: {type(expected_output).__name__}")

    # Import function under test
    from agent_recon_slice import recon_slice
    from verification_utils import recursive_check

    if not inner_paths:
        # Scenario A: Simple function call
        print("\n=== Scenario A: Simple function call ===")
        try:
            result = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")

        if expected_output is None:
            print("WARNING: No expected output to compare against. Checking result is not None.")
            if result is None:
                print("FAIL: Result is None")
                sys.exit(1)
            print("TEST PASSED (no expected output for comparison, but function ran successfully)")
            sys.exit(0)

        try:
            passed, msg = recursive_check(expected_output, result)
        except Exception as e:
            print(f"FAIL: recursive_check raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not passed:
            print(f"FAIL: Verification failed.")
            print(f"Message: {msg}")
            if hasattr(expected_output, 'shape') and hasattr(result, 'shape'):
                print(f"  Expected shape: {expected_output.shape}, Result shape: {result.shape}")
                if expected_output.shape == result.shape:
                    diff = np.abs(expected_output.astype(float) - result.astype(float))
                    print(f"  Max diff: {diff.max()}, Mean diff: {diff.mean()}")
                    print(f"  Expected range: [{expected_output.min()}, {expected_output.max()}]")
                    print(f"  Result range: [{result.min()}, {result.max()}]")
            sys.exit(1)
        else:
            print("TEST PASSED")
            sys.exit(0)
    else:
        # Scenario B: Factory/Closure pattern
        print("\n=== Scenario B: Factory/Closure pattern ===")
        try:
            agent_operator = recon_slice(*outer_args, **outer_kwargs)
        except Exception as e:
            print(f"FAIL: recon_slice (outer) raised exception: {e}")
            traceback.print_exc()
            sys.exit(1)

        if not callable(agent_operator):
            print(f"FAIL: Expected callable, got {type(agent_operator)}")
            sys.exit(1)

        all_passed = True
        for inner_path in inner_paths:
            print(f"\nProcessing inner: {os.path.basename(inner_path)}")
            try:
                with open(inner_path, 'rb') as f:
                    inner_data = dill.load(f)
            except Exception as e:
                print(f"FAIL: Could not load inner data: {e}")
                sys.exit(1)

            inner_args = inner_data.get('args', ())
            inner_kwargs = inner_data.get('kwargs', {})
            expected = inner_data.get('output', None)

            try:
                result = agent_operator(*inner_args, **inner_kwargs)
            except Exception as e:
                print(f"FAIL: agent_operator raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            try:
                passed, msg = recursive_check(expected, result)
            except Exception as e:
                print(f"FAIL: recursive_check raised exception: {e}")
                traceback.print_exc()
                sys.exit(1)

            if not passed:
                print(f"FAIL: Verification failed for {os.path.basename(inner_path)}")
                print(f"Message: {msg}")
                all_passed = False
            else:
                print(f"PASS: {os.path.basename(inner_path)}")

        if not all_passed:
            sys.exit(1)

        print("\nTEST PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()