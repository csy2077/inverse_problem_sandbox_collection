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
            out_dir = '/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/std_data'
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import root
import scipy.linalg
import timeit
import warnings
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    warnings.warn('tabulate not installed. Summary will be printed in basic format.')

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def load_data(json_path=None):
    """
    Load model data and configuration from JSON file.
    
    Parameters
    ----------
    json_path : str, optional
        Path to the JSON data file. If None, uses the default path based on script location.
    
    Returns
    -------
    dict
        Dictionary containing all model data and configuration.
    """
    if json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, 'data', 'standalone_mcmc_goodwin_oscillator.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Data file not found: {json_path}')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def get_suggested_times(data):
    """
    Generate suggested times array from data configuration.
    
    Parameters
    ----------
    data : dict
        Data dictionary loaded from JSON file.
    
    Returns
    -------
    np.ndarray
        Array of time points.
    """
    times_config = data['suggested_times']
    return np.linspace(times_config['start'], times_config['end'], times_config['n_points'])

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def vector(x):
    """
    Copies x and returns a 1d read-only NumPy array of floats with shape (n,).
    """
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

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def matrix2d(x):
    """
    Copies x and returns a 2d read-only NumPy array of floats with shape (m, n).
    """
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

    def format(self, time=None):
        """Formats a number of seconds into a readable string."""
        if time is None:
            time = self.time()
        if time < 0.01:
            return str(time) + ' seconds'
        elif time < 60:
            return str(round(time, 2)) + ' seconds'
        output = []
        time = int(round(time))
        units = [(604800, 'week'), (86400, 'day'), (3600, 'hour'), (60, 'minute')]
        for (k, name) in units:
            f = time // k
            if f > 0 or output:
                output.append(str(f) + ' ' + (name if f == 1 else name + 's'))
            time -= f * k
        output.append('1 second' if time == 1 else str(time) + ' seconds')
        return ', '.join(output)

    def reset(self):
        """Resets this timer's start time."""
        self._start = timeit.default_timer()

    def time(self):
        """Returns the time (in seconds) since this timer was created."""
        return timeit.default_timer() - self._start

class GoodwinOscillatorModel:
    """
    Three-state Goodwin oscillator toy model for gene expression oscillations.
    
    The model considers level of mRNA (x), which is translated into protein (y),
    which, in turn, stimulates production of protein (z) that inhibits production
    of mRNA. The ODE system is described by:

        dx/dt = 1 / (1 + z^10) - m1 * x
        dy/dt = k2 * x - m2 * y
        dz/dt = k3 * y - m3 * z

    Parameters: [k2, k3, m1, m2, m3]
    Outputs: 3 (x, y, z states)
    """

    def __init__(self, initial_conditions=None):
        """
        Initialize the Goodwin Oscillator model.
        
        Parameters
        ----------
        initial_conditions : list or np.ndarray, optional
            Initial conditions [x0, y0, z0]. If None, uses default values.
        """
        if initial_conditions is None:
            self._y0 = np.array([0.0054, 0.053, 1.93])
        else:
            self._y0 = np.array(initial_conditions)

    def n_outputs(self):
        """Returns the number of outputs (3 states)."""
        return 3

    def n_parameters(self):
        """Returns the number of parameters (5)."""
        return 5

    def _rhs(self, state, t, parameters):
        """Calculates the model RHS."""
        (x, y, z) = state
        (k2, k3, m1, m2, m3) = parameters
        dxdt = 1 / (1 + z ** 10) - m1 * x
        dydt = k2 * x - m2 * y
        dzdt = k3 * y - m3 * z
        return [dxdt, dydt, dzdt]

    def simulate(self, parameters, times):
        """Runs a forward simulation with the given parameters."""
        times = np.array(times)
        offset = 0
        if len(times) < 1 or times[0] != 0:
            times = np.concatenate(([0], times))
            offset = 1
        values = odeint(self._rhs, self._y0, times, args=(parameters,))
        return values[offset:, :self.n_outputs()]

class MultiOutputProblem:
    """
    Represents an inference problem where a model is fit to a multi-valued time series.
    """

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
            raise ValueError('Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """Runs a simulation using the given parameters."""
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def evaluateS1(self, parameters):
        """Runs a simulation with sensitivities (numerical approximation)."""
        y = self.evaluate(parameters)
        eps = 1e-06
        dy = np.zeros((self._n_times, self._n_outputs, self._n_parameters))
        for i in range(self._n_parameters):
            p_plus = np.array(parameters, copy=True)
            p_plus[i] += eps
            y_plus = self.evaluate(p_plus)
            dy[:, :, i] = (y_plus - y) / eps
        return (y, dy)

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

class RectangularBoundaries:
    """Represents rectangular boundaries on a parameter space."""

    def __init__(self, lower, upper):
        self._lower = vector(lower)
        self._upper = vector(upper)
        self._n_parameters = len(self._lower)
        if len(self._upper) != self._n_parameters:
            raise ValueError('Lower and upper bounds must have same length.')
        if np.any(self._upper <= self._lower):
            raise ValueError('Upper bounds must exceed lower bounds.')

    def n_parameters(self):
        return self._n_parameters

    def lower(self):
        return self._lower

    def upper(self):
        return self._upper

    def range(self):
        return self._upper - self._lower

    def check(self, x):
        """Check if x is within bounds."""
        return np.all(x >= self._lower) and np.all(x < self._upper)

    def sample(self, n=1):
        """Sample from uniform distribution within bounds."""
        return np.random.uniform(self._lower, self._upper, size=(n, self._n_parameters))

class UniformLogPrior:
    """
    Defines a uniform prior over a given range.
    """

    def __init__(self, lower, upper):
        self._boundaries = RectangularBoundaries(lower, upper)
        self._n_parameters = self._boundaries.n_parameters()
        self._value = -np.log(np.prod(self._boundaries.range()))

    def __call__(self, x):
        return self._value if self._boundaries.check(x) else -np.inf

    def evaluateS1(self, x):
        """Returns log-prior and gradient (zero for uniform prior)."""
        return (self(x), np.zeros(self._n_parameters))

    def n_parameters(self):
        return self._n_parameters

class GaussianKnownSigmaLogLikelihood:
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
        self._sigma = sigma
        self._offset = -0.5 * self._nt * np.log(2 * np.pi)
        self._offset -= self._nt * np.log(sigma)
        self._multip = -1 / (2.0 * sigma ** 2)
        self._isigma2 = sigma ** (-2)

    def __call__(self, x):
        error = self._values - self._problem.evaluate(x)
        return np.sum(self._offset + self._multip * np.sum(error ** 2, axis=0))

    def evaluateS1(self, x):
        """See LogPDF.evaluateS1()."""
        (y, dy) = self._problem.evaluateS1(x)
        dy = dy.reshape(self._nt, self._no, self._np)
        r = self._values - y
        L = np.sum(self._offset + self._multip * np.sum(r ** 2, axis=0))
        dL = np.sum((self._isigma2 * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)
        return (L, dL)

    def n_parameters(self):
        return self._np

class GaussianLogLikelihood:
    """
    Calculates a log-likelihood assuming independent Gaussian noise at each
    time point, and adds a parameter representing the standard deviation
    (sigma) of the noise on each output.
    """

    def __init__(self, problem):
        self._problem = problem
        self._values = problem.values()
        self._times = problem.times()
        self._nt = len(self._times)
        self._no = problem.n_outputs()
        self._n_parameters = problem.n_parameters() + self._no
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        error = self._values - self._problem.evaluate(x[:-self._no])
        return np.sum(-self._logn - self._nt * np.log(sigma) - np.sum(error ** 2, axis=0) / (2 * sigma ** 2))

    def evaluateS1(self, x):
        """See LogPDF.evaluateS1()."""
        sigma = np.asarray(x[-self._no:])
        L = self.__call__(x)
        if np.isneginf(L):
            return (L, np.tile(np.nan, self._n_parameters))
        (y, dy) = self._problem.evaluateS1(x[:-self._no])
        dy = dy.reshape(self._nt, self._no, self._n_parameters - self._no)
        r = self._values - y
        dL = np.sum((sigma ** (-2.0) * np.sum((r.T * dy.T).T, axis=0).T).T, axis=0)
        dsigma = -self._nt / sigma + sigma ** (-3.0) * np.sum(r ** 2, axis=0)
        dL = np.concatenate((dL, np.array(list(dsigma))))
        return (L, dL)

    def n_parameters(self):
        return self._n_parameters

class LogPosterior:
    """
    Represents a posterior distribution.
    Calculates: log(posterior) = log(likelihood) + log(prior)
    """

    def __init__(self, log_likelihood, log_prior):
        self._log_likelihood = log_likelihood
        self._log_prior = log_prior
        self._n_parameters = log_likelihood.n_parameters()

    def __call__(self, x):
        prior = self._log_prior(x)
        if np.isinf(prior):
            return prior
        return prior + self._log_likelihood(x)

    def evaluateS1(self, x):
        """Evaluate log-posterior and its gradient."""
        (prior, dprior) = self._log_prior.evaluateS1(x)
        if np.isinf(prior):
            return (prior, dprior)
        (likelihood, dlikelihood) = self._log_likelihood.evaluateS1(x)
        return (prior + likelihood, dprior + dlikelihood)

    def n_parameters(self):
        return self._n_parameters

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def autocorrelation(x):
    """Calculates autocorrelation for a vector x."""
    x = (x - np.mean(x)) / (np.std(x) * np.sqrt(len(x)))
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _autocorrelate_negative(autocorrelation):
    """Returns the index of the first negative entry in autocorrelation."""
    try:
        return np.where(np.asarray(autocorrelation) < 0)[0][0]
    except IndexError:
        return len(autocorrelation)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def effective_sample_size_single_parameter(x):
    """Calculates effective sample size (ESS) for a single parameter."""
    rho = autocorrelation(x)
    T = _autocorrelate_negative(rho)
    n = len(x)
    ess = n / (1 + 2 * np.sum(rho[0:T]))
    return ess

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def effective_sample_size(samples):
    """Calculates effective sample size (ESS) for n-dimensional samples."""
    try:
        (n_samples, n_params) = samples.shape
    except (ValueError, IndexError):
        raise ValueError('Samples must be given as a 2d array.')
    if n_samples < 2:
        raise ValueError('At least two samples must be given.')
    return [effective_sample_size_single_parameter(samples[:, i]) for i in range(0, n_params)]

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _within(chains):
    """Calculates mean within-chain variance."""
    within_chain_var = np.var(chains, axis=1, ddof=1)
    return np.mean(within_chain_var, axis=0)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def _between(chains):
    """Calculates mean between-chain variance."""
    n = chains.shape[1]
    within_chain_means = np.mean(chains, axis=1)
    between_chain_var = np.var(within_chain_means, axis=0, ddof=1)
    return n * between_chain_var

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def rhat(chains, warm_up=0.0):
    """
    Returns the convergence measure R-hat for MCMC chains.
    """
    if not (chains.ndim == 2 or chains.ndim == 3):
        raise ValueError('Dimension of chains must be 2 or 3.')
    if warm_up > 1 or warm_up < 0:
        raise ValueError('warm_up only takes values in [0,1].')
    n = chains.shape[1]
    chains = chains[:, int(n * warm_up):]
    n = chains.shape[1]
    n = n // 2
    if n < 1:
        raise ValueError('Number of samples per chain after warm-up is too small.')
    chains = np.vstack([chains[:, :n], chains[:, -n:]])
    w = _within(chains)
    b = _between(chains)
    return np.sqrt((n - 1.0) / n + b / (w * n))

class MCMCSummary:
    """
    Calculates and prints key summaries of posterior samples from MCMC chains.
    """

    def __init__(self, chains, time=None, parameter_names=None):
        self._chains = chains
        self._chains_unmodified = chains
        if len(chains) == 1:
            warnings.warn('Summaries calculated with one chain may be unreliable.')
        self._n_parameters = chains[0].shape[1]
        if time is not None and float(time) <= 0:
            raise ValueError('Elapsed time must be positive.')
        self._time = time
        if parameter_names is None:
            parameter_names = ['param ' + str(i + 1) for i in range(self._n_parameters)]
        elif self._n_parameters != len(parameter_names):
            raise ValueError('Parameter names list must be same length as number of sampled parameters')
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
            if HAS_TABULATE:
                self._summary_str = tabulate(self._summary_list, headers=headers, numalign='left', floatfmt='.2f')
            else:
                lines = [' | '.join(headers)]
                for row in self._summary_list:
                    lines.append(' | '.join(['{:.2f}'.format(x) if isinstance(x, float) else str(x) for x in row]))
                self._summary_str = '\n'.join(lines)
        return self._summary_str

    def _make_summary(self):
        """Calculates posterior summaries for all parameters."""
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
        for i in range(0, self._n_parameters):
            row = [self._parameter_names[i], self._mean[i], self._std[i], self._quantiles[0, i], self._quantiles[1, i], self._quantiles[2, i], self._quantiles[3, i], self._quantiles[4, i], self._rhat[i], self._ess[i]]
            if self._time is not None:
                row.append(self._ess_per_second[i])
            self._summary_list.append(row)

    def mean(self):
        return self._mean

    def std(self):
        return self._std

    def ess(self):
        return self._ess

class HaarioBardenetACMC:
    """
    Haario-Bardenet Adaptive Covariance MCMC method.
    A single-chain MCMC method with adaptive proposal covariance.
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

    def in_initial_phase(self):
        return not self._adaptive

    def needs_initial_phase(self):
        return True

    def needs_sensitivities(self):
        return False

    def set_initial_phase(self, initial_phase):
        self._adaptive = not bool(initial_phase)

    def acceptance_rate(self):
        return self._acceptance_rate

    def ask(self):
        if not self._running:
            self._running = True
            self._proposed = self._x0
            self._proposed.setflags(write=False)
        if self._proposed is None:
            self._proposed = np.random.multivariate_normal(self._current, self._sigma * np.exp(self._log_lambda))
            self._proposed.setflags(write=False)
        return self._proposed

    def tell(self, fx):
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

class RelativisticMCMC:
    """
    Relativistic Monte Carlo method.
    Uses relativistic Hamiltonian dynamics for efficient exploration.
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
        self._ready_for_tell = False
        self._current = None
        self._current_energy = None
        self._current_gradient = None
        self._current_momentum = None
        self._momentum = None
        self._position = None
        self._gradient = None
        self._mcmc_iteration = 0
        self._mcmc_acceptance = 0
        self._frog_iteration = 0
        self._n_frog_iterations = 20
        self._epsilon = 0.1
        self._step_size = None
        self._mass = 1
        self._c = 10
        self.set_leapfrog_step_size(np.diag(self._sigma0))
        self._divergent = np.asarray([], dtype='int')
        self._hamiltonian_threshold = 10 ** 3
        self._max_integration_size = 100000000.0

    def name(self):
        return 'Relativistic MCMC'

    def needs_initial_phase(self):
        return False

    def needs_sensitivities(self):
        return True

    def set_initial_phase(self, initial_phase):
        pass

    def set_leapfrog_step_size(self, step_size):
        """Sets the step size for the leapfrog algorithm."""
        a = np.atleast_1d(step_size)
        if len(a[a < 0]) > 0:
            raise ValueError('Step size must be greater than zero.')
        if len(a) == 1:
            step_size = np.repeat(step_size, self._n_parameters)
        elif not len(step_size) == self._n_parameters:
            raise ValueError('Step size should match number of parameters')
        self._step_size = step_size
        self._set_scaled_epsilon()

    def set_leapfrog_steps(self, steps):
        """Sets the number of leapfrog steps."""
        steps = int(steps)
        if steps < 1:
            raise ValueError('Number of steps must exceed 0.')
        self._n_frog_iterations = steps

    def _set_scaled_epsilon(self):
        """Rescales epsilon along dimensions of step_size."""
        self._scaled_epsilon = np.zeros(self._n_parameters)
        for i in range(self._n_parameters):
            self._scaled_epsilon[i] = self._epsilon * self._step_size[i]

    def _calculate_momentum_distribution(self):
        """Calculate an approximation to the CDF of momentum magnitude."""

        def logpdf_deriv(p):
            d = np.sqrt(self._mass * (self._n_parameters - 1)) * (p ** 2 + self._m2c2) ** 0.25 / np.sqrt(self._c * self._mass) - p
            return d
        p_max = root(logpdf_deriv, self._mass).x[0]
        max_value = 2 * p_max
        spacing = max_value / 1000
        integration_accepted = False
        while not integration_accepted:
            integration_grid = np.arange(min(1e-06, 0.5 * spacing), max_value, spacing)
            logpdf_values = self._momentum_logpdf(integration_grid)
            cdf = np.logaddexp.accumulate(np.logaddexp(logpdf_values[1:], logpdf_values[:-1]))
            cdf = np.exp(cdf - cdf[-1])
            if np.diff(cdf)[-1] > 0.001 * max(np.diff(cdf)):
                max_value *= 2
                spacing *= 2
            elif max(np.diff(cdf)) > 0.001:
                spacing /= 2
            else:
                integration_accepted = True
            if max_value / spacing > self._max_integration_size:
                warnings.warn('Failed to approximate momentum distribution.')
                integration_accepted = True
        inv_cdf = interp1d([0.0] + list(cdf), integration_grid)
        self._inv_cdf = inv_cdf

    def _momentum_logpdf(self, u):
        """Evaluate the unnormalized logpdf of the magnitude of momentum."""
        return -self._mc2 * np.sqrt(u ** 2 / self._m2c2 + 1) + np.log(u ** (self._n_parameters - 1))

    def _sample_momentum(self):
        """Draw a sample of the momentum vector."""
        dir = np.random.randn(self._n_parameters)
        dir /= np.linalg.norm(dir)
        u = np.random.random()
        p = self._inv_cdf(u)
        return p * dir

    def _kinetic_energy(self, momentum):
        """Kinetic energy of relativistic particle."""
        squared = np.sum(np.array(momentum) ** 2)
        return self._mc2 * (squared / self._m2c2 + 1) ** 0.5

    def ask(self):
        """Returns a parameter vector to evaluate."""
        if self._ready_for_tell:
            raise RuntimeError('Ask() called when expecting call to tell().')
        if not self._running:
            self._running = True
            self._mc2 = self._mass * self._c ** 2
            self._m2c2 = self._mass ** 2 * self._c ** 2
            self._calculate_momentum_distribution()
        if self._current is None:
            self._ready_for_tell = True
            return np.array(self._x0, copy=True)
        if self._frog_iteration == 0:
            self._current_momentum = self._sample_momentum()
            self._position = np.array(self._current, copy=True)
            self._gradient = np.array(self._current_gradient, copy=True)
            self._momentum = np.array(self._current_momentum, copy=True)
            self._momentum -= self._scaled_epsilon * self._gradient * 0.5
        squared = np.sum(np.array(self._momentum) ** 2)
        relativistic_mass = self._mass * np.sqrt(squared / self._m2c2 + 1)
        self._position += self._scaled_epsilon * self._momentum / relativistic_mass
        self._ready_for_tell = True
        return np.array(self._position, copy=True)

    def tell(self, reply):
        """Performs an iteration of the MCMC algorithm."""
        if not self._ready_for_tell:
            raise RuntimeError('Tell called before proposal was set.')
        self._ready_for_tell = False
        (energy, gradient) = reply
        energy = float(energy)
        gradient = vector(gradient)
        energy = -energy
        gradient = -gradient
        if self._current is None:
            if not np.isfinite(energy):
                raise ValueError('Initial point for MCMC must have finite logpdf.')
            self._current = self._x0
            self._current_energy = energy
            self._current_gradient = gradient
            self._mcmc_iteration += 1
            self._current.setflags(write=False)
            return (self._current, (-self._current_energy, -self._current_gradient), True)
        self._gradient = gradient
        self._frog_iteration += 1
        if self._frog_iteration < self._n_frog_iterations:
            self._momentum -= self._scaled_epsilon * self._gradient
            return None
        self._momentum -= self._scaled_epsilon * self._gradient * 0.5
        accept = 0
        if np.isfinite(energy) and np.all(np.isfinite(self._momentum)):
            current_U = self._current_energy
            current_K = self._kinetic_energy(self._current_momentum)
            proposed_U = energy
            proposed_K = self._kinetic_energy(self._momentum)
            div = proposed_U + proposed_K - (self._current_energy + current_K)
            if np.abs(div) > self._hamiltonian_threshold:
                self._divergent = np.append(self._divergent, self._mcmc_iteration)
                self._momentum = self._position = self._gradient = None
                self._frog_iteration = 0
                self._mcmc_iteration += 1
                self._mcmc_acceptance = (self._mcmc_iteration * self._mcmc_acceptance + accept) / (self._mcmc_iteration + 1)
                return (self._current, (-self._current_energy, -self._current_gradient), False)
            else:
                r = np.exp(current_U - proposed_U + current_K - proposed_K)
                if np.random.uniform(0, 1) < r:
                    accept = 1
                    self._current = self._position
                    self._current_energy = energy
                    self._current_gradient = gradient
                    self._current.setflags(write=False)
        self._momentum = self._position = self._gradient = None
        self._frog_iteration = 0
        self._mcmc_iteration += 1
        self._mcmc_acceptance = (self._mcmc_iteration * self._mcmc_acceptance + accept) / (self._mcmc_iteration + 1)
        return (self._current, (-self._current_energy, -self._current_gradient), accept != 0)

class MCMCController:
    """
    Samples from a LogPDF using MCMC.
    """

    def __init__(self, log_pdf, chains, x0, sigma0=None, method=None):
        self._log_pdf = log_pdf
        self._n_parameters = log_pdf.n_parameters()
        self._n_chains = int(chains)
        if self._n_chains < 1:
            raise ValueError('Number of chains must be at least 1.')
        if len(x0) != chains:
            raise ValueError('Number of initial positions must equal number of chains.')
        if not all([len(x) == self._n_parameters for x in x0]):
            raise ValueError('All initial positions must have same dimension as the LogPDF.')
        if method is None:
            method = HaarioBardenetACMC
        self._method = method
        self._samplers = [method(x, sigma0) for x in x0]
        self._needs_initial_phase = self._samplers[0].needs_initial_phase()
        self._needs_sensitivities = self._samplers[0].needs_sensitivities()
        self._initial_phase_iterations = 200 if self._needs_initial_phase else None
        self._log_to_screen = True
        self._log_interval = 500
        self._max_iterations = 10000
        self._parallel = False
        self._samples = None
        self._time = None

    def samplers(self):
        """Returns list of internal samplers."""
        return self._samplers

    def set_max_iterations(self, iterations=10000):
        if iterations is not None:
            iterations = int(iterations)
            if iterations < 0:
                raise ValueError('Maximum number of iterations cannot be negative.')
        self._max_iterations = iterations

    def set_log_to_screen(self, enabled):
        self._log_to_screen = True if enabled else False

    def set_log_interval(self, interval):
        self._log_interval = int(interval)

    def set_parallel(self, parallel):
        self._parallel = bool(parallel)

    def time(self):
        return self._time

    def run(self):
        """Runs the MCMC sampler and returns the chains."""
        if self._max_iterations is None:
            raise ValueError('At least one stopping criterion must be set.')
        iteration = 0
        if self._needs_initial_phase:
            for sampler in self._samplers:
                sampler.set_initial_phase(True)
        if self._log_to_screen:
            print('Using ' + str(self._samplers[0].name()))
            print('Generating ' + str(self._n_chains) + ' chains.')
            print('Running in sequential mode.')
        samples = np.zeros((self._n_chains, self._max_iterations, self._n_parameters))
        n_samples = [0] * self._n_chains
        active = list(range(self._n_chains))
        timer = Timer()
        running = True
        while running:
            if self._needs_initial_phase and iteration == self._initial_phase_iterations:
                for sampler in self._samplers:
                    sampler.set_initial_phase(False)
                if self._log_to_screen:
                    print('Initial phase completed.')
            xs = [self._samplers[i].ask() for i in active]
            if self._needs_sensitivities:
                fxs = [self._log_pdf.evaluateS1(x) for x in xs]
            else:
                fxs = [self._log_pdf(x) for x in xs]
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
            iteration += 1
            if iteration % self._log_interval == 0 and self._log_to_screen:
                print(f'Iteration {iteration}/{self._max_iterations}')
            if self._max_iterations is not None and iteration >= self._max_iterations:
                running = False
        self._time = timer.time()
        if self._log_to_screen:
            print(f'Halting: Maximum number of iterations ({iteration}) reached.')
            print(f'Time: {timer.format(self._time)}')
        self._samples = samples
        return samples

class XNES:
    """
    Exponential Natural Evolution Strategy optimizer.
    """

    def __init__(self, x0, sigma0=None):
        self._x0 = vector(x0)
        self._n_parameters = len(self._x0)
        if sigma0 is None:
            self._sigma0 = np.abs(self._x0)
            self._sigma0[self._sigma0 == 0] = 1
        else:
            self._sigma0 = np.array(sigma0, copy=True).flatten()
            if len(self._sigma0) == 1:
                self._sigma0 = np.repeat(self._sigma0, self._n_parameters)
        self._running = False
        self._mu = np.array(self._x0, copy=True)
        self._A = np.diag(self._sigma0)
        self._x_best = np.array(self._x0, copy=True)
        self._f_best = np.inf
        self._f_guessed = np.inf
        self._n_pop = 4 + int(3 * np.log(self._n_parameters))
        self._eta_mu = 1
        self._eta_sigma = None
        self._eta_B = None
        self._xs = None
        self._zs = None

    def name(self):
        return 'xNES'

    def ask(self):
        """Returns points to evaluate."""
        if not self._running:
            self._running = True
            n = self._n_parameters
            self._eta_mu = 1
            self._eta_sigma = (3 + np.log(n)) * (0.25 / (n * np.sqrt(n)))
            self._eta_B = (3 + np.log(n)) * (0.25 / (n * np.sqrt(n)))
        self._zs = np.random.normal(0, 1, (self._n_pop, self._n_parameters))
        self._xs = self._mu + np.dot(self._zs, self._A.T)
        return np.array(self._xs, copy=True)

    def tell(self, fxs):
        """Updates based on function evaluations."""
        fxs = np.array(fxs)
        order = np.argsort(fxs)
        if fxs[order[0]] < self._f_best:
            self._x_best = self._xs[order[0]]
            self._f_best = fxs[order[0]]
        n = self._n_pop
        utils = np.zeros(n)
        for (i, j) in enumerate(order):
            utils[j] = max(0, np.log(n / 2 + 1) - np.log(i + 1))
        utils = utils / np.sum(utils) - 1 / n
        g_delta = np.dot(utils, self._zs)
        g_M = np.zeros((self._n_parameters, self._n_parameters))
        for i in range(n):
            g_M += utils[i] * (np.outer(self._zs[i], self._zs[i]) - np.eye(self._n_parameters))
        self._mu = self._mu + self._eta_mu * np.dot(self._A, g_delta)
        self._A = np.dot(self._A, scipy.linalg.expm(0.5 * self._eta_sigma * np.trace(g_M) * np.eye(self._n_parameters) + 0.5 * self._eta_B * (g_M - np.trace(g_M) * np.eye(self._n_parameters))))

    def x_best(self):
        return self._x_best

    def f_best(self):
        return self._f_best

    def x_guessed(self):
        return self._mu

    def f_guessed(self):
        return self._f_guessed

class OptimisationController:
    """
    Controller for running optimization.
    """

    def __init__(self, function, x0, sigma0=None, method=None):
        self._function = function
        self._n_parameters = function.n_parameters()
        if method is None:
            method = XNES
        self._optimiser = method(x0, sigma0)
        self._max_iterations = 10000
        self._log_to_screen = True

    def set_max_iterations(self, iterations):
        self._max_iterations = int(iterations)

    def set_log_to_screen(self, enabled):
        self._log_to_screen = bool(enabled)

    def run(self):
        """Runs the optimization."""
        for iteration in range(self._max_iterations):
            xs = self._optimiser.ask()
            fxs = [-self._function(x) for x in xs]
            self._optimiser.tell(fxs)
            if iteration % 100 == 0 and self._log_to_screen:
                print(f'Iteration {iteration}, f_best: {-self._optimiser.f_best():.4f}')
        return (self._optimiser.x_best(), -self._optimiser.f_best())

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def plot_trace(samples, ref_parameters=None, parameter_names=None, save_path='mcmc_trace.png'):
    """
    Creates and saves trace plots and histograms for MCMC samples.
    """
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    (_, n_param) = samples[0].shape
    if parameter_names is None:
        parameter_names = ['Parameter' + str(i + 1) for i in range(n_param)]
    (fig, axes) = plt.subplots(n_param, 2, figsize=(12, 2 * n_param), squeeze=False)
    stacked_chains = np.vstack(samples)
    xmin = np.min(stacked_chains, axis=0)
    xmax = np.max(stacked_chains, axis=0)
    xbins = np.linspace(xmin, xmax, bins)
    for i in range(n_param):
        (ymin_all, ymax_all) = (np.inf, -np.inf)
        for (j_list, samples_j) in enumerate(samples):
            axes[i, 0].set_xlabel(parameter_names[i])
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].hist(samples_j[:, i], bins=xbins[:, i], alpha=alpha, label='Chain ' + str(1 + j_list))
            axes[i, 1].set_xlabel('Iteration')
            axes[i, 1].set_ylabel(parameter_names[i])
            axes[i, 1].plot(samples_j[:, i], alpha=alpha)
            ymin_all = min(ymin_all, xmin[i])
            ymax_all = max(ymax_all, xmax[i])
        axes[i, 1].set_ylim([ymin_all, ymax_all])
        if ref_parameters is not None and i < len(ref_parameters):
            axes[i, 0].axvline(ref_parameters[i], color='k', linestyle='--', label='True')
            axes[i, 1].axhline(ref_parameters[i], color='k', linestyle='--')
    if n_list > 1:
        axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Trace plot saved to: {save_path}')
    return (fig, axes)

@_record_io_decorator_(save_path='/fs-computility-new/UPDZ02_sunhe/shared/QA_yixuan/standalone_mcmc_goodwin_oscillator_sandbox/run_code/meta_data.json')
@_data_capture_decorator_
def plot_series(samples, problem, save_path='posterior_predictive.png', thinning=None, state_names=None):
    """
    Creates and saves posterior predictive plots.
    """
    try:
        (n_sample, n_param) = samples.shape
    except ValueError:
        raise ValueError('samples must be of shape (n_sample, n_parameters).')
    n_parameters = problem.n_parameters()
    n_outputs = problem.n_outputs()
    if thinning is None:
        thinning = max(1, int(n_sample / 200))
    else:
        thinning = int(thinning)
        if thinning < 1:
            raise ValueError('Thinning rate must be None or an integer > 0.')
    times = problem.times()
    predicted_values = []
    for params in samples[::thinning, :n_parameters]:
        predicted_values.append(problem.evaluate(params))
    predicted_values = np.array(predicted_values)
    mean_values = np.mean(predicted_values, axis=0)
    alpha = min(1, max(0.05 * (1000 / (n_sample / thinning)), 0.5))
    (fig, axes) = plt.subplots(n_outputs, 1, figsize=(10, 3 * n_outputs), sharex=True)
    if n_outputs == 1:
        axes = [axes]
    if state_names is None:
        state_names = ['mRNA (x)', 'Protein (y)', 'Protein (z)']
    for i_output in range(n_outputs):
        axes[i_output].set_ylabel(state_names[i_output] if i_output < len(state_names) else f'Output {i_output + 1}')
        axes[i_output].plot(times, problem.values()[:, i_output], 'x', color='#7f7f7f', ms=6.5, alpha=0.5, label='Observed data')
        axes[i_output].plot(times, predicted_values[0][:, i_output], color='#1f77b4', label='Inferred series')
        for v in predicted_values[1:]:
            axes[i_output].plot(times, v[:, i_output], color='#1f77b4', alpha=alpha)
        axes[i_output].plot(times, mean_values[:, i_output], 'k:', lw=2, label='Mean of inferred series')
    axes[0].legend()
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Posterior predictive plot saved to: {save_path}')
    return (fig, axes)
if __name__ == '__main__':
    print('=' * 70)
    print('Goodwin Oscillator Model - MCMC Inference')
    print('=' * 70)
    print('\nLoading data from JSON file...')
    data = load_data()
    print(f"Loaded model: {data['model']['name']}")
    model_config = data['model']
    initial_conditions = model_config['initial_conditions']
    model = GoodwinOscillatorModel(initial_conditions=initial_conditions)
    real_parameters = np.array(data['suggested_parameters'])
    times = get_suggested_times(data)
    print('\nTrue Parameters:')
    param_names = model_config['parameter_names']
    print(f'  {param_names[0]}: {real_parameters[0]}, {param_names[1]}: {real_parameters[1]}')
    print(f'  {param_names[2]}: {real_parameters[2]}, {param_names[3]}: {real_parameters[3]}, {param_names[4]}: {real_parameters[4]}')
    values = model.simulate(real_parameters, times)
    state_names = model_config['state_names']
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(times, values[:, 0], 'b')
    plt.ylabel(state_names[0])
    plt.title('Goodwin Oscillator - Clean Simulation')
    plt.subplot(3, 1, 2)
    plt.plot(times, values[:, 1], 'g')
    plt.ylabel(state_names[1])
    plt.subplot(3, 1, 3)
    plt.plot(times, values[:, 2], 'r')
    plt.ylabel(state_names[2])
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('goodwin_simulation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nClean simulation plot saved to: goodwin_simulation.png')
    noise_config = data['noise_levels']
    noise1 = noise_config['sigma_x']
    noise2 = noise_config['sigma_y']
    noise3 = noise_config['sigma_z']
    noisy_values = np.array(values, copy=True)
    np.random.seed(data['random_seed'])
    noisy_values[:, 0] += np.random.normal(0, noise1, len(times))
    noisy_values[:, 1] += np.random.normal(0, noise2, len(times))
    noisy_values[:, 2] += np.random.normal(0, noise3, len(times))
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(times, noisy_values[:, 0], 'b')
    plt.ylabel(state_names[0])
    plt.title('Goodwin Oscillator - Noisy Data')
    plt.subplot(3, 1, 2)
    plt.plot(times, noisy_values[:, 1], 'g')
    plt.ylabel(state_names[1])
    plt.subplot(3, 1, 3)
    plt.plot(times, noisy_values[:, 2], 'r')
    plt.ylabel(state_names[2])
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('goodwin_noisy_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Noisy data plot saved to: goodwin_noisy_data.png')
    print('\n' + '=' * 70)
    print('PART 1: MCMC with Known Noise Levels')
    print('=' * 70)
    mcmc1_config = data['mcmc_settings']['part1_known_noise']
    problem = MultiOutputProblem(model, times, values)
    log_prior = UniformLogPrior(mcmc1_config['prior_lower'], mcmc1_config['prior_upper'])
    log_likelihood = GaussianKnownSigmaLogLikelihood(problem, [noise1, noise2, noise3])
    log_posterior = LogPosterior(log_likelihood, log_prior)
    x0 = mcmc1_config['initial_positions'] * mcmc1_config['n_chains']
    mcmc = MCMCController(log_posterior, mcmc1_config['n_chains'], x0)
    mcmc.set_max_iterations(mcmc1_config['max_iterations'])
    mcmc.set_log_to_screen(False)
    print(f"\nRunning MCMC ({mcmc1_config['max_iterations']} iterations, {mcmc1_config['n_chains']} chains)...")
    print(f"(Originally {mcmc1_config['original_iterations']} iterations - reduced 10x for speed)")
    chains = mcmc.run()
    print('Done!')
    results = MCMCSummary(chains=chains, time=mcmc.time(), parameter_names=mcmc1_config['parameter_names'])
    print('\nMCMC Summary (Known Noise):')
    print(results)
    plot_trace(chains, ref_parameters=real_parameters, parameter_names=mcmc1_config['parameter_names'], save_path='goodwin_mcmc_trace_1.png')
    print('\n' + '=' * 70)
    print('PART 2: Optimization with XNES')
    print('=' * 70)
    opt_config = data['mcmc_settings']['optimization']
    opt = OptimisationController(log_posterior, opt_config['initial_position'], method=XNES)
    opt.set_log_to_screen(False)
    opt.set_max_iterations(opt_config['max_iterations'])
    (parameters_opt, fbest) = opt.run()
    print('\nOptimization Results:')
    print('            ' + '       '.join(mcmc1_config['parameter_names']))
    print('real  ' + ' '.join(['{: 8.4g}'.format(float(x)) for x in real_parameters]))
    print('found ' + ' '.join(['{: 8.4g}'.format(x) for x in parameters_opt]))
    print('\n' + '=' * 70)
    print('PART 3: Relativistic MCMC with Unknown Noise Levels')
    print('=' * 70)
    mcmc3_config = data['mcmc_settings']['part3_unknown_noise']
    problem2 = MultiOutputProblem(model, times, noisy_values)
    log_likelihood2 = GaussianLogLikelihood(problem2)
    log_prior2 = UniformLogPrior(mcmc3_config['prior_lower'], mcmc3_config['prior_upper'])
    log_posterior2 = LogPosterior(log_likelihood2, log_prior2)
    real_parameters_with_noise = np.array(list(real_parameters) + [noise1, noise2, noise3])
    xs = [real_parameters_with_noise * m for m in mcmc3_config['initial_position_multipliers']]
    mcmc2 = MCMCController(log_posterior2, mcmc3_config['n_chains'], xs, method=RelativisticMCMC)
    mcmc2.set_max_iterations(mcmc3_config['max_iterations'])
    mcmc2.set_log_to_screen(True)
    mcmc2.set_log_interval(1)
    for sampler in mcmc2.samplers():
        sampler.set_leapfrog_step_size(mcmc3_config['leapfrog_step_size'])
        sampler.set_leapfrog_steps(mcmc3_config['leapfrog_steps'])
    print(f"\nRunning Relativistic MCMC ({mcmc3_config['max_iterations']} iterations, {mcmc3_config['n_chains']} chains)...")
    print(f"(Originally {mcmc3_config['original_iterations']} iterations - reduced 10x for speed)")
    chains2 = mcmc2.run()
    print('Done!')
    results2 = MCMCSummary(chains=chains2, time=mcmc2.time(), parameter_names=mcmc3_config['parameter_names'])
    print('\nMCMC Summary (Unknown Noise):')
    print(results2)
    plot_trace(chains2, ref_parameters=real_parameters_with_noise, parameter_names=mcmc3_config['parameter_names'], save_path='goodwin_mcmc_trace_2.png')
    stacked_chains = np.vstack(chains2)
    plot_series(stacked_chains, problem2, save_path='goodwin_series.png', state_names=state_names)
    print('\n' + '=' * 70)
    print('Done!')
    print('=' * 70)
    print('\nOutput files:')
    print('  - goodwin_simulation.png: Clean model simulation')
    print('  - goodwin_noisy_data.png: Noisy data used for inference')
    print('  - goodwin_mcmc_trace_1.png: MCMC trace plots (known noise)')
    print('  - goodwin_mcmc_trace_2.png: MCMC trace plots (unknown noise)')
    print('  - goodwin_series.png: Posterior predictive time series')