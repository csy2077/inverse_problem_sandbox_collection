import os as _os_
import sys as _sys_
import functools as _functools_
import dill as _dill_
import time as _time_
import inspect as _inspect_
import json as _json_
_META_REGISTRY_ = set()
try:
    import numpy as _np_
except ImportError:
    _np_ = None
try:
    import torch as _torch_
except ImportError:
    _torch_ = None

def _fix_seeds_(seed=42):
    import random
    if _np_:
        _np_.random.seed(seed)
    random.seed(seed)
    if _torch_:
        _torch_.manual_seed(seed)
        if _torch_.cuda.is_available():
            _torch_.cuda.manual_seed_all(seed)
_fix_seeds_(42)

def _analyze_obj_(obj):
    if _torch_ and isinstance(obj, _torch_.Tensor):
        return {'type': 'torch.Tensor', 'shape': list(obj.shape), 'dtype': str(obj.dtype), 'device': str(obj.device)}
    if _np_ and isinstance(obj, _np_.ndarray):
        return {'type': 'numpy.ndarray', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return {'type': type(obj).__name__, 'length': len(obj), 'elements': [_analyze_obj_(item) for item in obj]}
    if hasattr(obj, '__dict__'):
        methods = []
        try:
            for m in dir(obj):
                if m.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, m)
                    if callable(attr):
                        methods.append(m)
                except Exception:
                    continue
        except Exception:
            pass
        return {'type': 'CustomObject', 'class_name': obj.__class__.__name__, 'public_methods': methods, 'attributes': list(obj.__dict__.keys())}
    try:
        val_str = str(obj)
    except:
        val_str = '<non-stringifiable>'
    return {'type': type(obj).__name__, 'value_sample': val_str}

def _record_io_decorator_(save_path='./'):

    def decorator(func, parent_function=None):

        @_functools_.wraps(func)
        def wrapper(*args, **kwargs):
            global _META_REGISTRY_
            func_name = func.__name__
            parent_key = str(parent_function)
            registry_key = (func_name, parent_key)
            should_record = False
            if registry_key not in _META_REGISTRY_:
                should_record = True
            result = None
            inputs_meta = {}
            if should_record:
                try:
                    sig = _inspect_.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for (name, value) in bound_args.arguments.items():
                        inputs_meta[name] = _analyze_obj_(value)
                except Exception as e:
                    inputs_meta = {'error': f'Analysis failed: {e}'}
            result = func(*args, **kwargs)
            if should_record:
                try:
                    output_meta = _analyze_obj_(result)
                except Exception:
                    output_meta = 'Analysis failed'
                try:
                    final_path = save_path
                    if not final_path.endswith('.json'):
                        if not _os_.path.exists(final_path):
                            _os_.makedirs(final_path, exist_ok=True)
                        if parent_function == None:
                            final_path = _os_.path.join(final_path, f'IO_meta_{func_name}.json')
                        else:
                            final_path = _os_.path.join(final_path, f'IO_meta_parent_function_{parent_function}_{func_name}.json')
                    dir_name = _os_.path.dirname(final_path)
                    if dir_name and (not _os_.path.exists(dir_name)):
                        _os_.makedirs(dir_name, exist_ok=True)
                    existing_data = []
                    file_exists = _os_.path.exists(final_path)
                    if file_exists:
                        try:
                            with open(final_path, 'r') as f:
                                existing_data = _json_.load(f)
                        except:
                            pass
                    already_in_file = False
                    for entry in existing_data:
                        if entry.get('function_name') == func_name:
                            already_in_file = True
                            break
                    if not already_in_file:
                        func_schema = {'function_name': func_name, 'inputs': inputs_meta, 'output': output_meta}
                        existing_data.append(func_schema)
                        with open(final_path, 'w') as f:
                            _json_.dump(existing_data, f, indent=4)
                        print(f'  [Metadata] Recorded schema for: {func_name}')
                    _META_REGISTRY_.add(registry_key)
                except Exception as e:
                    print(f'  [Metadata] Warning: {e}')
            if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
                return decorator(result, parent_function=func_name)
            return result
        return wrapper
    return decorator

def _data_capture_decorator_(func, parent_function=None):

    @_functools_.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/std_data'
            if not _os_.path.exists(out_dir):
                _os_.makedirs(out_dir, exist_ok=True)
            func_name = func.__name__
            if parent_function == None:
                save_path = _os_.path.join(out_dir, f'data_{func_name}.pkl')
            else:
                save_path = _os_.path.join(out_dir, f'data_parent_{parent_function}_{func_name}.pkl')

            def detach_recursive(obj):
                if hasattr(obj, 'detach'):
                    return obj.detach()
                if isinstance(obj, list):
                    return [detach_recursive(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple((detach_recursive(x) for x in obj))
                if isinstance(obj, dict):
                    return {k: detach_recursive(v) for (k, v) in obj.items()}
                return obj
            payload = {'func_name': func_name, 'args': detach_recursive(args), 'kwargs': detach_recursive(kwargs), 'output': detach_recursive(result)}
            with open(save_path, 'wb') as f:
                _dill_.dump(payload, f)
        except Exception as e:
            pass
        if callable(result) and (not isinstance(result, type)) and _inspect_.isfunction(result):
            return _data_capture_decorator_(result, parent_function=func_name)
        return result
    return wrapper
import os
import json
import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import warnings
from tabulate import tabulate

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def load_data(json_file):
    """
    Load model data and parameters from a JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing model data.
    
    Returns
    -------
    dict
        Dictionary containing all loaded data.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    data['model_initial_conditions'] = np.array(data['model_initial_conditions'])
    data['suggested_parameters'] = np.array(data['suggested_parameters'])
    data['suggested_times'] = np.array(data['suggested_times'])
    data['mcmc_initial_guesses'] = [list(x) for x in data['mcmc_initial_guesses']]
    return data
FLOAT_FORMAT = '{: .17e}'

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def strfloat(x):
    """Converts a float to a string, with maximum precision."""
    return FLOAT_FORMAT.format(float(x))

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def vector(x):
    """Copies x and returns a 1d read-only NumPy array of floats with shape (n,)."""
    if np.isscalar(x):
        x = np.array([float(x)])
    else:
        x = np.array(x, copy=True, dtype=float)
    x.setflags(write=False)
    if x.ndim != 1:
        n = np.max(x.shape)
        if np.prod(x.shape) != n:
            raise ValueError('Unable to convert to 1d vector of scalar values.')
        x = x.reshape((n,))
    return x

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def matrix2d(x):
    """Copies x and returns a 2d read-only NumPy array of floats."""
    x = np.array(x, copy=True, dtype=float)
    if x.ndim == 1:
        x = x.reshape((len(x), 1))
    elif x.ndim != 2:
        raise ValueError('Unable to convert to 2d matrix.')
    x.setflags(write=False)
    return x

class Timer:
    """Provides accurate timing."""

    def __init__(self):
        self._start = timeit.default_timer()

    def time(self):
        return timeit.default_timer() - self._start

    def reset(self):
        self._start = timeit.default_timer()

class RepressilatorModel:
    """
    The Repressilator model describes oscillations in a network of proteins
    that suppress their own creation.
    
    The formulation has three protein states (p_i) and three mRNA states (m_i):
    
    dm_0/dt = -m_0 + alpha/(1 + p_2^n) + alpha_0
    dm_1/dt = -m_1 + alpha/(1 + p_0^n) + alpha_0
    dm_2/dt = -m_2 + alpha/(1 + p_1^n) + alpha_0
    dp_0/dt = -beta * (p_0 - m_0)
    dp_1/dt = -beta * (p_1 - m_1)
    dp_2/dt = -beta * (p_2 - m_2)
    
    With parameters: alpha_0, alpha, beta, n
    Only the mRNA states are visible as output.
    """

    def __init__(self, y0=None):
        if y0 is None:
            self._y0 = np.array([0, 0, 0, 2, 1, 3])
        else:
            self._y0 = np.array(y0, dtype=float)
            if len(self._y0) != 6:
                raise ValueError('Initial value must have size 6.')
            if np.any(self._y0 < 0):
                raise ValueError('Initial states can not be negative.')

    def n_outputs(self):
        """Returns the number of outputs (3 mRNA states)."""
        return 3

    def n_parameters(self):
        """Returns the number of parameters (4: alpha_0, alpha, beta, n)."""
        return 4

    def _rhs(self, y, t, alpha_0, alpha, beta, n):
        """Calculates the model RHS."""
        dy = np.zeros(6)
        dy[0] = -y[0] + alpha / (1 + y[5] ** n) + alpha_0
        dy[1] = -y[1] + alpha / (1 + y[3] ** n) + alpha_0
        dy[2] = -y[2] + alpha / (1 + y[4] ** n) + alpha_0
        dy[3] = -beta * (y[3] - y[0])
        dy[4] = -beta * (y[4] - y[1])
        dy[5] = -beta * (y[5] - y[2])
        return dy

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        (alpha_0, alpha, beta, n) = parameters
        y = odeint(self._rhs, self._y0, times, (alpha_0, alpha, beta, n))
        return y[:, :3]

class MultiOutputProblem:
    """Represents an inference problem with multiple outputs."""

    def __init__(self, model, times, values):
        self._model = model
        self._times = vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be non-decreasing.')
        self._values = matrix2d(values)
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError('Values array must have shape (n_times, n_outputs).')

    def evaluate(self, parameters):
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def n_outputs(self):
        return self._n_outputs

    def n_parameters(self):
        return self._n_parameters

    def n_times(self):
        return self._n_times

    def times(self):
        return self._times

    def values(self):
        return self._values

class LogPDF:
    """Represents the natural logarithm of a probability density function."""

    def __call__(self, x):
        raise NotImplementedError

    def n_parameters(self):
        raise NotImplementedError

class GaussianKnownSigmaLogLikelihood(LogPDF):
    """
    Calculates a log-likelihood assuming independent Gaussian noise at each
    time point, using a known value for the standard deviation (sigma).
    """

    def __init__(self, problem, sigma):
        self._problem = problem
        self._values = problem.values()
        self._times = problem.times()
        self._no = problem.n_outputs()
        self._np = problem.n_parameters()
        self._nt = problem.n_times()
        if np.isscalar(sigma):
            sigma = np.ones(self._no) * float(sigma)
        else:
            sigma = vector(sigma)
            if len(sigma) != self._no:
                raise ValueError('Sigma must be a scalar or a vector of length n_outputs.')
        if np.any(sigma <= 0):
            raise ValueError('Standard deviation must be greater than zero.')
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma ** 2)
        self._n_parameters = self._np

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error ** 2, axis=0))

    def n_parameters(self):
        return self._n_parameters

class SequentialEvaluator:
    """Evaluates function sequentially."""

    def __init__(self, function):
        self._function = function

    def evaluate(self, positions):
        return [self._function(x) for x in positions]

class ParallelEvaluator:
    """Evaluates function in parallel."""

    def __init__(self, function, n_workers=None):
        import multiprocessing
        self._function = function
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        self._n_workers = n_workers

    def evaluate(self, positions):
        import multiprocessing
        with multiprocessing.Pool(self._n_workers) as pool:
            return pool.map(self._function, positions)

    @staticmethod
    def cpu_count():
        import multiprocessing
        return multiprocessing.cpu_count()

class Logger:
    """Basic logging class."""

    def __init__(self):
        self._stream = True
        self._filename = None
        self._csv = False
        self._fields = []
        self._line = []
        self._header_logged = False

    def add_counter(self, name, max_value=None):
        self._fields.append(('counter', name, max_value))

    def add_float(self, name):
        self._fields.append(('float', name))

    def add_time(self, name):
        self._fields.append(('time', name))

    def log(self, *args):
        for arg in args:
            self._line.append(arg)
        if len(self._line) >= len(self._fields):
            self._write_line()
            self._line = []

    def set_stream(self, stream):
        self._stream = stream

    def set_filename(self, filename, csv=False):
        self._filename = filename
        self._csv = csv

    def _write_line(self):
        if not self._header_logged:
            if self._stream:
                header = ' '.join([f[1] for f in self._fields])
                print(header)
            self._header_logged = True
        if self._stream:
            parts = []
            for (i, (ftype, name, *rest)) in enumerate(self._fields):
                val = self._line[i]
                if ftype == 'counter':
                    parts.append(f'{val:>8}')
                elif ftype == 'float':
                    parts.append(f'{val:>12.4e}')
                elif ftype == 'time':
                    parts.append(f'{val:>10.2f}')
            print(' '.join(parts))

class HaarioBardenetACMC:
    """
    Haario-Bardenet Adaptive Covariance MCMC sampler.
    
    This is an adaptive Metropolis algorithm that adapts the proposal
    covariance matrix based on the history of samples.
    """

    def __init__(self, x0, sigma0=None):
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)
        if sigma0 is None:
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
            self._sigma0 = np.diag(0.01 * self._sigma0)
        else:
            self._sigma0 = np.array(sigma0, copy=True)
            if np.prod(self._sigma0.shape) == self._n_parameters:
                self._sigma0 = self._sigma0.reshape((self._n_parameters,))
                self._sigma0 = np.diag(self._sigma0)
            else:
                self._sigma0 = self._sigma0.reshape((self._n_parameters, self._n_parameters))
        self._running = False
        self._adaptive = False
        self._current = None
        self._current_log_pdf = None
        self._proposed = None
        self._iterations = 0
        self._adaptations = 1
        self._target_acceptance = 0.234
        self._acceptance_count = 0
        self._acceptance_rate = 0
        self._mu = np.array(self._x0, copy=True)
        self._sigma = np.array(self._sigma0, copy=True)
        self._eta = 0.6
        self._gamma = 1
        self._log_lambda = 0

    def name(self):
        return 'Haario-Bardenet adaptive covariance MCMC'

    def needs_sensitivities(self):
        return False

    def needs_initial_phase(self):
        return True

    def in_initial_phase(self):
        return not self._adaptive

    def set_initial_phase(self, initial_phase):
        self._adaptive = not bool(initial_phase)

    def ask(self):
        """Returns a parameter vector to evaluate the LogPDF for."""
        if not self._running:
            self._running = True
            self._proposed = self._x0
            self._proposed.setflags(write=False)
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(self._current, self._sigma * np.exp(self._log_lambda))
            self._proposed.setflags(write=False)
        return self._proposed

    def tell(self, fx):
        """
        Performs an iteration of the MCMC algorithm.
        Returns (current position, current log pdf, accepted).
        """
        if self._proposed is None:
            raise RuntimeError('Tell called before proposal was set.')
        fx = float(fx)
        self._iterations += 1
        if self._current is None:
            if not np.isfinite(fx):
                raise ValueError('Initial point for MCMC must have finite logpdf.')
            self._current = self._proposed
            self._current_log_pdf = fx
            self._proposed = None
            return (self._current, self._current_log_pdf, True)
        log_ratio = fx - self._current_log_pdf
        accepted = False
        if np.isfinite(fx):
            u = np.log(np.random.uniform(0, 1))
            if u < log_ratio:
                accepted = True
                self._acceptance_count += 1
                self._current = self._proposed
                self._current_log_pdf = fx
        self._acceptance_rate = self._acceptance_count / self._iterations
        self._proposed = None
        if self._adaptive:
            self._gamma = (self._adaptations + 1) ** (-self._eta)
            self._adaptations += 1
            self._mu = (1 - self._gamma) * self._mu + self._gamma * self._current
            dsigm = np.reshape(self._current - self._mu, (self._n_parameters, 1))
            self._sigma = (1 - self._gamma) * self._sigma + self._gamma * np.dot(dsigm, dsigm.T)
            p = 1 if accepted else 0
            self._log_lambda += self._gamma * (p - self._target_acceptance)
        return (self._current, self._current_log_pdf, accepted)

    def _log_init(self, logger):
        logger.add_float('Accept.')

    def _log_write(self, logger):
        logger.log(self._acceptance_rate)

class MCMCController:
    """Controls MCMC sampling."""

    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()
        self._n_chains = int(chains)
        if len(x0) != chains:
            raise ValueError('Number of initial positions must equal number of chains.')
        if method is None:
            method = HaarioBardenetACMC
        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        self._log_to_screen = True
        self._max_iterations = 10000
        self._parallel = False
        self._n_workers = 1
        self._message_interval = 20
        self._message_warm_up = 3
        self._initial_phase_iterations = 200
        self._has_run = False
        self._samples = None
        self._time = None

    def set_max_iterations(self, iterations):
        self._max_iterations = int(iterations)

    def set_log_interval(self, iters=20, warm_up=3):
        self._message_interval = int(iters)
        self._message_warm_up = max(0, int(warm_up))

    def set_log_to_screen(self, enabled):
        self._log_to_screen = True if enabled else False

    def set_parallel(self, parallel=False):
        if parallel is True:
            self._parallel = True
            self._n_workers = ParallelEvaluator.cpu_count()
        elif parallel >= 1:
            self._parallel = True
            self._n_workers = int(parallel)
        else:
            self._parallel = False
            self._n_workers = 1

    def set_initial_phase_iterations(self, iterations=200):
        self._initial_phase_iterations = int(iterations)

    def samplers(self):
        return self._samplers

    def time(self):
        return self._time

    def run(self):
        if self._has_run:
            raise RuntimeError('Controller is valid for single use only')
        self._has_run = True
        f = self._log_pdf
        if self._parallel:
            n_workers = min(self._n_workers, self._n_chains)
            evaluator = ParallelEvaluator(f, n_workers=n_workers)
        else:
            evaluator = SequentialEvaluator(f)
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)
        samples = np.zeros((self._n_chains, self._max_iterations, self._n_parameters))
        active = list(range(self._n_chains))
        n_samples = [0] * self._n_chains
        if self._log_to_screen:
            print('Using ' + str(self._samplers[0].name()))
            print('Generating ' + str(self._n_chains) + ' chains.')
            if self._parallel:
                print('Running in parallel.')
            else:
                print('Running in sequential mode.')
        logger = Logger()
        if not self._log_to_screen:
            logger.set_stream(None)
        logger.add_counter('Iter.', max_value=self._max_iterations)
        logger.add_counter('Eval.', max_value=self._max_iterations * self._n_chains)
        for sampler in self._samplers:
            sampler._log_init(logger)
        logger.add_time('Time')
        timer = Timer()
        iteration = 0
        n_evaluations = 0
        next_message = 0
        while active:
            if self._needs_initial_phase and iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')
            xs = [self._samplers[i].ask() for i in active]
            fxs = evaluator.evaluate(xs)
            n_evaluations += len(fxs)
            fxs_iterator = iter(fxs)
            for i in list(active):
                reply = self._samplers[i].tell(next(fxs_iterator))
                if reply is not None:
                    (y, fy, accepted) = reply
                    samples[i][n_samples[i]] = y
                    n_samples[i] += 1
                    if n_samples[i] == self._max_iterations:
                        active.remove(i)
            intermediate_step = min(n_samples) <= iteration
            if intermediate_step:
                continue
            if iteration >= next_message:
                logger.log(iteration, n_evaluations)
                for sampler in self._samplers:
                    sampler._log_write(logger)
                logger.log(timer.time())
                if iteration < self._message_warm_up:
                    next_message = iteration + 1
                else:
                    next_message = self._message_interval * (1 + iteration // self._message_interval)
            iteration += 1
        self._time = timer.time()
        self._samples = samples
        print(f'Halting: Maximum number of iterations ({self._max_iterations}) reached.')
        return samples

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def autocorrelation(x):
    """Calculates autocorrelation for a vector x."""
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _autocorrelate_negative(autocorrelation):
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def effective_sample_size_single_parameter(x):
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def effective_sample_size(samples):
    try:
        (n_samples, n_params) = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')
    return [effective_sample_size_single_parameter(samples[:, i]) for i in range(n_params)]

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _within(chains):
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _between(chains):
    n = chains.shape[1]
    within_chain_means = np.mean(chains, axis=1)
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)
    return n * between_chain_var

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def rhat(chains, warm_up=0.0):
    """Returns the convergence measure R-hat."""
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError('Chains must be 2 or 3 dimensional.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError('warm_up must be in [0,1].')
    n = chains.shape[1]
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]
    n = n // 2
    if n < 1:
        raise ValueError('Not enough samples.')
    chains = np.vstack([chains[:, :n], chains[:, -n:]])
    w = _within(chains)
    b = _between(chains)
    return np.sqrt((n - 1.0) / n + b / (w * n))

class MCMCSummary:
    """Calculates and prints key summaries of posterior samples."""

    def __init__(self, chains, time=None, parameter_names=None):
        self._chains = chains
        self._chains_unmodified = chains
        if len(chains) == 1:
            warnings.warn('Summaries with one chain may be unreliable.')
        self._n_parameters = chains[0].shape[1]
        if time is not None and float(time) <= 0:
            raise ValueError('Elapsed time must be positive.')
        self._time = time
        if parameter_names is None:
            parameter_names = ['param ' + str(i + 1) for i in range(self._n_parameters)]
        elif self._n_parameters != len(parameter_names):
            raise ValueError('Parameter names list must match number of parameters.')
        self._parameter_names = parameter_names
        self._ess = None
        self._ess_per_second = None
        self._mean = None
        self._quantiles = None
        self._rhat = None
        self._std = None
        self._summary_list = []
        self._summary_str = None
        self._make_summary()

    def __str__(self):
        if self._summary_str is None:
            headers = ['param', 'mean', 'std.', '2.5%', '25%', '50%', '75%', '97.5%', 'rhat', 'ess']
            if self._time is not None:
                headers.append('ess per sec.')
            self._summary_str = tabulate(self._summary_list, headers=headers, numalign='left', floatfmt='.2f')
        return self._summary_str

    def _make_summary(self):
        stacked = np.vstack(self._chains)
        self._mean = np.mean(stacked, axis=0)
        self._std = np.std(stacked, axis=0)
        self._quantiles = np.percentile(stacked, [2.5, 25, 50, 75, 97.5], axis=0)
        self._rhat = rhat(self._chains)
        self._ess = np.zeros(self._n_parameters)
        for (i, chain) in enumerate(self._chains):
            self._ess += effective_sample_size(chain)
        if self._time is not None:
            self._ess_per_second = np.array(self._ess) / self._time
        for i in range(self._n_parameters):
            row = [self._parameter_names[i], self._mean[i], self._std[i], self._quantiles[0, i], self._quantiles[1, i], self._quantiles[2, i], self._quantiles[3, i], self._quantiles[4, i], self._rhat[i], self._ess[i]]
            if self._time is not None:
                row.append(self._ess_per_second[i])
            self._summary_list.append(row)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def plot_trace(samples, n_percentiles=None, parameter_names=None, ref_parameters=None, filename=None):
    """Creates trace plots for MCMC samples."""
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    (_, n_param) = samples[0].shape
    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]
    (fig, axes) = plt.subplots(n_param, 2, figsize=(12, 2 * n_param), squeeze=False)
    stacked_chains = np.vstack(samples)
    if n_percentiles is None:
        xmin = np.min(stacked_chains, axis=0)
        xmax = np.max(stacked_chains, axis=0)
    else:
        xmin = np.percentile(stacked_chains, 50 - n_percentiles / 2.0, axis=0)
        xmax = np.percentile(stacked_chains, 50 + n_percentiles / 2.0, axis=0)
    xbins = np.linspace(xmin, xmax, bins)
    for i in range(n_param):
        (ymin_all, ymax_all) = (np.inf, -np.inf)
        for (j_list, samples_j) in enumerate(samples):
            axes[i, 0].set_xlabel(parameter_names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].hist(samples_j[:, i], bins=xbins[:, i], alpha=alpha, label='Samples ' + str(1 + j_list))
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel(parameter_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=alpha)
            ymin_all = min(ymin_all, xmin[i])
            ymax_all = max(ymax_all, xmax[i])
        axes[i, 1].set_ylim([ymin_all, ymax_all])
        if ref_parameters is not None:
            (ymin_tv, ymax_tv) = axes[i, 0].get_ylim()
            axes[i, 0].plot([ref_parameters[i], ref_parameters[i]], [0.0, ymax_tv], '--', c='k')
            (xmin_tv, xmax_tv) = axes[i, 1].get_xlim()
            axes[i, 1].plot([0.0, xmax_tv], [ref_parameters[i], ref_parameters[i]], '--', c='k')
    if n_list > 1:
        axes[0, 0].legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f'Saved: {filename}')
    return (fig, axes)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_repressilator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def plot_series(samples, problem, ref_parameters=None, thinning=None, filename=None):
    """Creates predicted time series plots."""
    try:
        (n_sample, n_param) = samples.shape
    except ValueError:
        raise ValueError('samples must be of shape (n_sample, n_parameters).')
    n_parameters = problem.n_parameters()
    n_outputs = problem.n_outputs()
    if ref_parameters is not None:
        ref_series = problem.evaluate(ref_parameters[:n_parameters])
    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning must be > 0.')
    times = problem.times()
    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)
    alpha = min(1, max(0.05 * (1000 / (n_sample / thinning)), 0.5))
    (fig, axes) = plt.subplots(n_outputs, 1, figsize=(8, np.sqrt(n_outputs) * 3), sharex=True)
    if n_outputs == 1:
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.plot(times, problem.values(), 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Original data')
        plt.plot(times, predicted_values[0], color='#1f77b4', label='Inferred series')
        for v in predicted_values[1:]:
            plt.plot(times, v, color='#1f77b4', alpha=alpha)
        plt.plot(times, mean_values, 'k:', lw=2, label='Mean of inferred series')
        if ref_parameters is not None:
            plt.plot(times, ref_series, color='#d62728', ls='--', label='Reference series')
        plt.legend()
    else:
        fig.subplots_adjust(hspace=0)
        axes[-1].set_xlabel('Time')
        for i_output in range(n_outputs):
            axes[i_output].set_ylabel('Output %d' % (i_output + 1))
            axes[i_output].plot(times, problem.values()[:, i_output], 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Original data')
            axes[i_output].plot(times, predicted_values[0][:, i_output], color='#1f77b4', label='Inferred series')
            for v in predicted_values[1:]:
                axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
            axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2, label='Mean of inferred series')
            if ref_parameters is not None:
                axes[i_output].plot(times, ref_series[:, i_output], color='#d62728', ls='--', label='Reference series')
        axes[0].legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f'Saved: {filename}')
    return (fig, axes)
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, 'data', 'standalone_mcmc_repressilator.json')
    print('=' * 60)
    print('Repressilator Model: MCMC Inference')
    print('=' * 60)
    print(f'\nLoading data from: {json_file}')
    data = load_data(json_file)
    np.random.seed(data['random_seed'])
    parameters = data['suggested_parameters']
    times = data['suggested_times']
    initial_conditions = data['model_initial_conditions']
    sigma = data['noise_sigma']
    x0 = data['mcmc_initial_guesses']
    max_iterations = data['mcmc_max_iterations']
    parameter_names = data['parameter_names']
    output_names = data['output_names']
    model = RepressilatorModel(y0=initial_conditions)
    values = model.simulate(parameters, times)
    print('\n--- Simulation ---')
    print('Parameters:')
    for (i, name) in enumerate(parameter_names):
        print(f'  {name} = {parameters[i]}')
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.plot(times, values)
    plt.legend(output_names)
    plt.title('Repressilator Model Simulation')
    plt.savefig('repressilator_simulation.png', dpi=150)
    plt.close()
    print('Saved: repressilator_simulation.png')
    noisy = values + np.random.normal(0, sigma, values.shape)
    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.plot(times, noisy)
    plt.legend(output_names)
    plt.title('Noisy Observations')
    plt.savefig('repressilator_noisy.png', dpi=150)
    plt.close()
    print('Saved: repressilator_noisy.png')
    problem = MultiOutputProblem(model, times, noisy)
    loglikelihood = GaussianKnownSigmaLogLikelihood(problem, sigma)
    print('\n--- MCMC Sampling ---')
    mcmc = MCMCController(loglikelihood, 3, x0)
    mcmc.set_log_to_screen(False)
    mcmc.set_max_iterations(max_iterations)
    print(f'Running MCMC with {max_iterations} iterations (reduced from 6000 for ~10x speedup)...')
    chains = mcmc.run()
    print('Done!')
    print('\n--- MCMC Summary ---')
    results = MCMCSummary(chains=chains, time=mcmc.time(), parameter_names=parameter_names)
    print(results)
    plt.figure()
    plot_trace(chains, ref_parameters=parameters, parameter_names=parameter_names, filename='repressilator_trace.png')
    plt.close()
    samples = chains[1][-100:]
    plt.figure(figsize=(12, 6))
    plot_series(samples, problem, filename='repressilator_series.png')
    plt.close()
    print('\n' + '=' * 60)
    print('All outputs saved successfully!')
    print('=' * 60)